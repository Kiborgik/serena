import threading
import time
from unittest.mock import MagicMock

import pytest

from serena.task_executor import TaskExecutor
from serena.tools.tools_base import Tool, ToolMarkerDoesNotRequireActiveProject


class _EchoTool(Tool, ToolMarkerDoesNotRequireActiveProject):
    def apply(self) -> str:
        return "done"


class _CacheSavingLanguageServerManager:
    def __init__(self, save_duration: float, fails: bool = False) -> None:
        self._save_duration = save_duration
        self._fails = fails
        self.saved = threading.Event()

    def save_all_caches_if_due(self) -> None:
        time.sleep(self._save_duration)
        self.saved.set()
        if self._fails:
            raise OSError("disk full")


@pytest.fixture
def executor() -> TaskExecutor:
    return TaskExecutor("ToolCacheSaveTestExecutor")


def _tool(executor: TaskExecutor, ls_manager: _CacheSavingLanguageServerManager) -> _EchoTool:
    agent = MagicMock()
    agent.issue_task = executor.issue_task
    agent.serena_config.tool_timeout = 10
    agent.get_language_server_manager.return_value = ls_manager
    return _EchoTool(agent)


def test_result_is_returned_before_a_slow_cache_save_finishes(executor: TaskExecutor) -> None:
    ls_manager = _CacheSavingLanguageServerManager(save_duration=1.0)

    started = time.monotonic()
    result = _tool(executor, ls_manager).apply_ex(log_call=False, catch_exceptions=False)

    assert result == "done"
    assert time.monotonic() - started < 0.5
    assert ls_manager.saved.wait(timeout=5)


def test_failing_cache_save_affects_neither_the_result_nor_later_calls(executor: TaskExecutor) -> None:
    ls_manager = _CacheSavingLanguageServerManager(save_duration=0, fails=True)
    tool = _tool(executor, ls_manager)

    assert tool.apply_ex(log_call=False, catch_exceptions=False) == "done"
    assert ls_manager.saved.wait(timeout=5)
    assert tool.apply_ex(log_call=False, catch_exceptions=False) == "done"
