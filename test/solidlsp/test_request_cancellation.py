# SPDX-License-Identifier: MIT

import json
import logging
import threading
from queue import Queue
from unittest.mock import MagicMock

import pytest

from solidlsp.ls_config import LanguageServerId
from solidlsp.ls_exceptions import SolidLSPException
from solidlsp.ls_process import StdioLanguageServer
from solidlsp.util.cancellation import CancellationToken


def _server_without_responses() -> tuple[StdioLanguageServer, MagicMock]:
    server = StdioLanguageServer(
        process_launch_info=MagicMock(),
        ls_id=LanguageServerId.PYTHON,
        determine_log_level=lambda _line: logging.INFO,
    )
    process = MagicMock()
    server._process = process
    return server, process


def _sent_messages(process: MagicMock) -> list[dict]:
    return [json.loads(call.args[0][-1]) for call in process.stdin.writelines.call_args_list]


def test_request_under_cancelled_token_is_not_sent() -> None:
    server, process = _server_without_responses()
    token = CancellationToken()
    token.cancel()

    with token.activate(), pytest.raises(SolidLSPException):
        server.send_request("textDocument/hover", {})

    assert _sent_messages(process) == []


def test_cancelling_abandons_the_pending_request_and_tells_the_server() -> None:
    server, process = _server_without_responses()
    token = CancellationToken()
    request_sent = threading.Event()
    process.stdin.writelines.side_effect = lambda _message: request_sent.set()
    outcome: Queue[type[Exception]] = Queue()

    def send_request() -> None:
        with token.activate():
            try:
                server.send_request("textDocument/hover", {})
            except Exception as e:
                outcome.put(type(e))

    thread = threading.Thread(target=send_request)
    thread.start()
    assert request_sent.wait(timeout=5)
    token.cancel()
    thread.join(timeout=5)

    assert outcome.get_nowait() is SolidLSPException
    request, cancel_notification = _sent_messages(process)
    assert cancel_notification["method"] == "$/cancelRequest"
    assert cancel_notification["params"] == {"id": request["id"]}


def test_request_outside_any_token_is_sent_as_before() -> None:
    server, process = _server_without_responses()
    request_sent = threading.Event()
    process.stdin.writelines.side_effect = lambda _message: request_sent.set()

    threading.Thread(target=lambda: server.send_request("textDocument/hover", {}), daemon=True).start()

    assert request_sent.wait(timeout=5)
    assert [message["method"] for message in _sent_messages(process)] == ["textDocument/hover"]
