from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import threading

import pytest

from jetank.app.signals import TERMINATION_SIGNALS, raise_on_termination_signals
from jetank.errors import TerminationRequested


def signal_self(signum: signal.Signals) -> None:
    os.kill(os.getpid(), signum)
    signal.pause()  # not reached if the handler already ran


@pytest.mark.parametrize("signum", TERMINATION_SIGNALS)
def test_signal_raises_and_unwinds_with_blocks(signum: signal.Signals) -> None:
    cleanups: list[str] = []

    def guarded_work() -> None:
        with raise_on_termination_signals():
            try:
                signal_self(signum)
            finally:
                cleanups.append("motors stopped")

    with pytest.raises(TerminationRequested, match=signum.name) as excinfo:
        guarded_work()
    assert cleanups == ["motors stopped"]
    assert excinfo.value.signal_name == signum.name


def test_is_not_swallowed_by_except_exception() -> None:
    assert not issubclass(TerminationRequested, Exception)


def test_previous_handlers_are_restored() -> None:
    before = {signum: signal.getsignal(signum) for signum in TERMINATION_SIGNALS}
    with raise_on_termination_signals():
        pass
    assert {signum: signal.getsignal(signum) for signum in TERMINATION_SIGNALS} == before


def test_handlers_are_restored_after_termination() -> None:
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(TerminationRequested), raise_on_termination_signals():
        signal_self(signal.SIGTERM)
    assert signal.getsignal(signal.SIGTERM) == before


def test_outside_main_thread_raises_value_error() -> None:
    errors: list[BaseException] = []

    def enter() -> None:
        try:
            with raise_on_termination_signals():
                pass
        except ValueError as exc:
            errors.append(exc)

    thread = threading.Thread(target=enter)
    thread.start()
    thread.join()
    assert len(errors) == 1


def test_sighup_to_a_real_process_runs_its_cleanup() -> None:
    """End to end: what happens to a robot process when its SSH session drops."""
    script = textwrap.dedent(
        """
        import signal, sys
        from jetank.app.signals import raise_on_termination_signals
        with raise_on_termination_signals():
            try:
                print("ready", flush=True)
                signal.pause()
            finally:
                print("motors stopped", flush=True)
        """
    )
    with subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as process:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        process.send_signal(signal.SIGHUP)
        stdout, stderr = process.communicate(timeout=10)
    assert "motors stopped" in stdout
    assert "TerminationRequested: received SIGHUP" in stderr
    assert process.returncode != 0
