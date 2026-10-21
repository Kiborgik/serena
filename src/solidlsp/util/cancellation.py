# SPDX-License-Identifier: MIT

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import ClassVar


class CancellationToken:
    """
    A signal that the work running while the token is active (see `activate`) is no longer wanted.
    Language server requests issued while a cancelled token is active are not sent, and pending ones are abandoned.
    """

    _current: ClassVar[ContextVar["CancellationToken | None"]] = ContextVar("solidlsp_cancellation_token", default=None)

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._is_cancelled = False
        self._callbacks: dict[int, Callable[[], None]] = {}
        self._next_callback_id = 0

    @classmethod
    def current(cls) -> "CancellationToken | None":
        """
        :return: the token that is active in the current context, if any
        """
        return cls._current.get()

    @contextmanager
    def activate(self) -> Iterator[None]:
        """
        Makes this token the active one in the current context for the duration of the `with` block.
        """
        reset_token = self._current.set(self)
        try:
            yield
        finally:
            self._current.reset(reset_token)

    def is_cancelled(self) -> bool:
        """
        :return: whether the token has been cancelled
        """
        with self._lock:
            return self._is_cancelled

    def cancel(self) -> None:
        """
        Cancels the token and calls the registered callbacks; cancelling again has no effect.
        """
        with self._lock:
            if self._is_cancelled:
                return
            self._is_cancelled = True
            callbacks = list(self._callbacks.values())
            self._callbacks.clear()
        for callback in callbacks:
            callback()

    def on_cancel(self, callback: Callable[[], None]) -> Callable[[], None]:
        """
        Registers a callback to be called when the token is cancelled.
        If the token is already cancelled, the callback is called right away.

        :param callback: the callback
        :return: a function that unregisters the callback
        """
        with self._lock:
            if not self._is_cancelled:
                callback_id = self._next_callback_id
                self._next_callback_id += 1
                self._callbacks[callback_id] = callback
                return lambda: self._remove_callback(callback_id)
        callback()
        return lambda: None

    def _remove_callback(self, callback_id: int) -> None:
        with self._lock:
            self._callbacks.pop(callback_id, None)
