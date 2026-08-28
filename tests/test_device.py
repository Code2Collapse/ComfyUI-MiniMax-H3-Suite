import pytest
import torch

from mmx_utils.device import run_with_cpu_fallback


def test_succeeds_first_call_no_fallback():
    calls: list[torch.device] = []

    def work(dev: torch.device) -> int:
        calls.append(dev)
        return 42

    out = run_with_cpu_fallback(work, device=torch.device("cpu"), label="t")
    assert out == 42
    assert len(calls) == 1
    assert calls[0].type == "cpu"


def test_runtime_oom_falls_back_to_cpu():
    calls: list[torch.device] = []

    def work(dev: torch.device) -> str:
        calls.append(dev)
        if dev.type != "cpu":
            raise RuntimeError("CUDA out of memory")
        return "ok"

    out = run_with_cpu_fallback(work, device=torch.device("cuda"), label="t")
    assert out == "ok"
    assert len(calls) == 2
    assert calls[0].type == "cuda"
    assert calls[1].type == "cpu"


def test_not_implemented_falls_back_to_cpu():
    calls: list[torch.device] = []

    def work(dev: torch.device) -> int:
        calls.append(dev)
        if dev.type != "cpu":
            raise NotImplementedError("op missing on MPS")
        return 7

    out = run_with_cpu_fallback(work, device=torch.device("mps"), label="t")
    assert out == 7
    assert len(calls) == 2
    assert calls[1].type == "cpu"


def test_logic_error_not_retried():
    calls: list[torch.device] = []

    def work(dev: torch.device) -> None:
        calls.append(dev)
        raise ValueError("bad shape")

    with pytest.raises(ValueError, match="bad shape"):
        run_with_cpu_fallback(work, device=torch.device("cuda"), label="t")
    assert len(calls) == 1


def test_cpu_failure_propagates():
    def work(dev: torch.device) -> None:
        if dev.type != "cpu":
            raise RuntimeError("CUDA out of memory")
        raise RuntimeError("cpu also failed")

    with pytest.raises(RuntimeError, match="cpu also failed"):
        run_with_cpu_fallback(work, device=torch.device("cuda"), label="t")


def test_cuda_oom_type_falls_back():
    calls: list[torch.device] = []

    def work(dev: torch.device) -> int:
        calls.append(dev)
        if dev.type != "cpu":
            raise torch.cuda.OutOfMemoryError("boom")
        return 1

    out = run_with_cpu_fallback(work, device=torch.device("cuda"), label="t")
    assert out == 1
    assert len(calls) == 2
