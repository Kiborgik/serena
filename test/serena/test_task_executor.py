import logging
import time
from concurrent.futures import CancelledError
from queue import Queue
from unittest.mock import MagicMock

import pytest

from serena.task_executor import TaskExecutor
from solidlsp.ls_config import LanguageServerId
from solidlsp.ls_exceptions import SolidLSPException
from solidlsp.ls_process import StdioLanguageServer


@pytest.fixture
def executor():
    """
    Fixture for a basic SerenaAgent without a project
    """
    return TaskExecutor("TestExecutor")


class Task:
    def __init__(self, delay: float, exception: bool = False):
        self.delay = delay
        self.exception = exception
        self.did_run = False

    def run(self):
        self.did_run = True
        time.sleep(self.delay)
        if self.exception:
            raise ValueError("Task failed")
        return True


def test_task_executor_sequence(executor):
    """
    Tests that a sequence of tasks is executed correctly
    """
    future1 = executor.issue_task(Task(1).run, name="task1")
    future2 = executor.issue_task(Task(1).run, name="task2")
    assert future1.result() is True
    assert future2.result() is True


def test_task_executor_exception(executor):
    """
    Tests that tasks that raise exceptions are handled correctly, i.e. that
      * the exception is propagated,
      * subsequent tasks are still executed.
    """
    future1 = executor.issue_task(Task(1, exception=True).run, name="task1")
    future2 = executor.issue_task(Task(1).run, name="task2")
    have_exception = False
    try:
        assert future1.result()
    except Exception as e:
        assert isinstance(e, ValueError)
        have_exception = True
    assert have_exception
    assert future2.result() is True


def test_task_executor_cancel_current(executor):
    """
    Tests that tasks that are cancelled are handled correctly, i.e. that
      * subsequent tasks are executed as soon as cancellation ensues.
      * the cancelled task raises CancelledError when result() is called.
    """
    start_time = time.time()
    future1 = executor.issue_task(Task(10).run, name="task1")
    future2 = executor.issue_task(Task(1).run, name="task2")
    time.sleep(1)
    future1.cancel()
    assert future2.result() is True
    end_time = time.time()
    assert (end_time - start_time) < 9, "Cancelled task did not stop in time"
    have_cancelled_error = False
    try:
        future1.result()
    except Exception as e:
        assert e.__class__.__name__ == "CancelledError"
        have_cancelled_error = True
    assert have_cancelled_error


def test_task_executor_cancel_future(executor):
    """
    Tests that when a future task is cancelled, it is never run at all
    """
    task1 = Task(10)
    task2 = Task(1)
    future1 = executor.issue_task(task1.run, name="task1")
    future2 = executor.issue_task(task2.run, name="task2")
    time.sleep(1)
    future2.cancel()
    future1.cancel()
    try:
        future2.result()
    except:
        pass
    assert task1.did_run
    assert not task2.did_run


def test_task_executor_cancellation_via_task_info(executor):
    start_time = time.time()
    executor.issue_task(Task(10).run, "task1")
    executor.issue_task(Task(10).run, "task2")
    task_infos = executor.get_current_tasks()
    task_infos2 = executor.get_current_tasks()

    # test expected tasks
    assert len(task_infos) == 2
    assert "task1" in task_infos[0].name
    assert "task2" in task_infos[1].name

    # test task identifiers being stable
    assert task_infos2[0].task_id == task_infos[0].task_id

    # test cancellation
    task_infos[0].cancel()
    time.sleep(0.5)
    task_infos3 = executor.get_current_tasks()
    assert len(task_infos3) == 1  # Cancelled task is gone from the queue
    task_infos3[0].cancel()
    try:
        task_infos3[0].future.result()
    except:
        pass
    end_time = time.time()
    assert (end_time - start_time) < 9, "Cancelled task did not stop in time"


def test_queued_task_timeout_counts_from_its_start(executor):
    executor.issue_task(Task(0.7).run, name="task1")
    future2 = executor.issue_task(Task(0.7).run, name="task2")

    assert future2.result_after_start(timeout=1.0) is True


def test_task_running_longer_than_timeout_times_out(executor):
    future = executor.issue_task(Task(1.0).run, name="task1")

    with pytest.raises(TimeoutError):
        future.result_after_start(timeout=0.3)


def test_task_cancelled_in_queue_raises_without_waiting_for_its_turn(executor):
    executor.issue_task(Task(3).run, name="task1")
    future2 = executor.issue_task(Task(1).run, name="task2")
    future2.cancel()

    start_time = time.time()
    with pytest.raises(CancelledError):
        future2.result_after_start(timeout=5)
    assert (time.time() - start_time) < 1


def test_idle_executor_starts_an_issued_task_without_delay(executor):
    delays = []
    for _ in range(10):
        time.sleep(0.02)
        issued_at = time.perf_counter()
        started_at = executor.issue_task(time.perf_counter, name="timestamp").result()
        delays.append(started_at - issued_at)

    assert max(delays) < 0.03


def test_timed_out_task_stops_at_its_pending_language_server_request(executor):
    server = StdioLanguageServer(
        process_launch_info=MagicMock(),
        ls_id=LanguageServerId.PYTHON,
        determine_log_level=lambda _line: logging.INFO,
    )
    server._process = MagicMock()
    outcome: Queue[type[Exception]] = Queue()

    def request_without_response() -> None:
        try:
            server.send_request("textDocument/hover", {})
        except Exception as e:
            outcome.put(type(e))
            raise

    future = executor.issue_task(request_without_response, name="blocked")
    with pytest.raises(TimeoutError):
        future.result_after_start(timeout=0.3)

    assert outcome.get(timeout=1) is SolidLSPException
    assert executor.issue_task(lambda: True, name="next").result(timeout=1) is True
