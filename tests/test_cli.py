from __future__ import annotations

from adk_meta_harness.cli import main
from adk_meta_harness.task_executor import EvalOutput, EvalResult


def test_optimize_temporal_starts_workflow(monkeypatch, tmp_path, capsys):
    dataset = tmp_path / "tasks"
    harness = tmp_path / "harness"
    dataset.mkdir()
    harness.mkdir()

    async def fake_start_optimize_workflow(optimize_input, **kwargs):
        assert optimize_input.dataset == str(dataset)
        assert optimize_input.initial_harness == str(harness)
        assert kwargs["server_url"] == "localhost:7233"
        assert kwargs["task_queue"] == "amh-tasks"
        return "wf-123", "run-456"

    monkeypatch.setattr(
        "adk_meta_harness.runner.temporal_runner.start_optimize_workflow",
        fake_start_optimize_workflow,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "amh",
            "optimize",
            "--dataset",
            str(dataset),
            "--initial-harness",
            str(harness),
            "--runner",
            "temporal",
        ],
    )

    main()

    out = capsys.readouterr().out
    assert "Started Temporal optimization workflow" in out
    assert "Workflow ID: wf-123" in out
    assert "Run ID: run-456" in out


def test_worker_command_runs_temporal_worker(monkeypatch, capsys):
    called: dict[str, str] = {}

    async def fake_run_worker(*, server_url: str, task_queue: str):
        called["server_url"] = server_url
        called["task_queue"] = task_queue

    monkeypatch.setattr("adk_meta_harness.runner.temporal_runner.run_worker", fake_run_worker)
    monkeypatch.setattr(
        "sys.argv",
        [
            "amh",
            "worker",
            "--server",
            "127.0.0.1:7233",
            "--task-queue",
            "custom-queue",
        ],
    )

    main()

    assert called == {
        "server_url": "127.0.0.1:7233",
        "task_queue": "custom-queue",
    }
    out = capsys.readouterr().out
    assert "Starting Temporal worker on 127.0.0.1:7233" in out


def test_eval_temporal_uses_local_eval_and_prints_note(monkeypatch, tmp_path, capsys):
    candidate = tmp_path / "candidate"
    dataset = tmp_path / "tasks"
    candidate.mkdir()
    dataset.mkdir()

    class _FakeRunner:
        async def evaluate(self, **kwargs):
            return EvalOutput(search_results=[EvalResult(task_name="task", passed=True, score=1.0)])

    def fake_get_runner(name: str, **kwargs):
        assert name == "temporal"
        assert kwargs == {}
        return _FakeRunner()

    monkeypatch.setattr("adk_meta_harness.runner.get_runner", fake_get_runner)
    monkeypatch.setattr("adk_meta_harness.judge.get_judge", lambda *_a, **_k: object())
    monkeypatch.setattr(
        "sys.argv",
        [
            "amh",
            "eval",
            "--candidate",
            str(candidate),
            "--dataset",
            str(dataset),
            "--runner",
            "temporal",
            "--server",
            "127.0.0.1:7233",
            "--task-queue",
            "queue-a",
        ],
    )

    main()

    out = capsys.readouterr().out
    assert "eval with --runner temporal currently executes locally" in out
