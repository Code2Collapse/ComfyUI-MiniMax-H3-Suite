"""A29 background jobs for the HDR dataset + training nodes (owner order A29, design docs/hdr/A29_DESIGN.md).

Long work (downloads, pair building, training) runs off ComfyUI's execution thread:

- ``start_thread(kind, title, fn, key=...)``: ``fn(job)`` runs in a daemon thread; it reports with
  ``job.set_progress`` / ``job.log`` and calls ``job.check_cancel()`` between items (raises ``JobCancelled``).
- ``start_process(kind, title, argv, ...)``: a subprocess (the trainer). stdout + stderr are merged, split on
  ``\\r`` and ``\\n`` (tqdm redraws with ``\\r``), appended to a log file and fed to an optional ``parse(job, line)``.
  Cancel kills the whole process tree (Windows: ``taskkill /T /F``; POSIX: the process group).
- ``wait_for(job, progress=, interrupted=)``: a node that WAITS polls the job, forwards its progress to ComfyUI's
  ProgressBar and cancels the job when ComfyUI's interrupt is pressed.
- ``StateFile``: a small JSON file inside the work folder recording finished items, so a re-run skips them.
- Routes (registered once, only when the server is up): ``GET /mmx/hdr/jobs``, ``GET /mmx/hdr/jobs/{id}/log``,
  ``POST /mmx/hdr/jobs/{id}/cancel``.

The registry lives in memory (one ComfyUI process); what must survive a restart is on disk (StateFile, log files,
ai-toolkit checkpoints).
"""
from __future__ import annotations

import collections
import itertools
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Callable

LOG = logging.getLogger("MiniMaxSuite.hdr_jobs")

LOG_TAIL_LINES = 2000                      # kept in memory per job (the file keeps everything)
LOSS_POINTS_MAX = 20000
STATES = ("queued", "running", "done", "error", "cancelled")
ACTIVE = ("queued", "running")


class JobCancelled(Exception):
    """Raised inside a job when Cancel was pressed (or ComfyUI was interrupted while a node waited)."""


class Job:
    _ids = itertools.count(1)

    def __init__(self, kind: str, title: str, key: str | None = None):
        self.id = f"{kind}-{int(time.time())}-{next(self._ids)}"
        self.kind, self.title, self.key = kind, title, key or ""
        self.status = "queued"
        self.done, self.total = 0, 0
        self.message = ""
        self.result: dict | None = None
        self.error = ""
        self.started = time.time()
        self.finished: float | None = None
        self.log_path: str | None = None
        self.loss: list[tuple[int, float]] = []
        self.returncode: int | None = None
        self._lines: collections.deque = collections.deque(maxlen=LOG_TAIL_LINES)
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.proc: subprocess.Popen | None = None

    # ---- reporting (called from the worker) ----
    def log(self, line: str) -> None:
        line = str(line).rstrip("\r\n")
        with self._lock:
            self._lines.append(line)

    def set_progress(self, done: int, total: int | None = None, message: str | None = None) -> None:
        with self._lock:
            self.done = int(done)
            if total is not None:
                self.total = int(total)
            if message is not None:
                self.message = str(message)

    def add_loss(self, step: int, loss: float) -> None:
        with self._lock:
            if self.loss and self.loss[-1][0] == step:
                self.loss[-1] = (step, loss)
            else:
                self.loss.append((step, loss))
                if len(self.loss) > LOSS_POINTS_MAX:
                    del self.loss[: len(self.loss) - LOSS_POINTS_MAX]

    def check_cancel(self) -> None:
        if self._cancel.is_set():
            raise JobCancelled(f"{self.title}: cancelled")

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    # ---- control ----
    def cancel(self) -> None:
        self._cancel.set()
        if self.proc is not None and self.proc.poll() is None:
            kill_tree(self.proc)

    def is_active(self) -> bool:
        return self.status in ACTIVE

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # ---- views ----
    def tail(self, n: int = 200) -> list[str]:
        with self._lock:
            return list(self._lines)[-max(0, int(n)):]

    def summary(self) -> dict:
        with self._lock:
            return {"id": self.id, "kind": self.kind, "title": self.title, "status": self.status, "done": self.done,
                    "total": self.total, "message": self.message, "error": self.error, "started": self.started,
                    "finished": self.finished, "log_path": self.log_path, "returncode": self.returncode,
                    "last_loss": self.loss[-1] if self.loss else None, "result": self.result}


_JOBS: "collections.OrderedDict[str, Job]" = collections.OrderedDict()
_REG_LOCK = threading.Lock()
MAX_JOBS_KEPT = 50


def _register(job: Job) -> None:
    with _REG_LOCK:
        _JOBS[job.id] = job
        while len(_JOBS) > MAX_JOBS_KEPT:
            old = next((k for k, j in _JOBS.items() if not j.is_active()), None)
            if old is None:
                break
            _JOBS.pop(old)


def get(job_id: str) -> Job | None:
    with _REG_LOCK:
        return _JOBS.get(job_id)


def jobs() -> list[Job]:
    with _REG_LOCK:
        return list(_JOBS.values())


def active_with_key(key: str) -> Job | None:
    """The running job doing the same work (same kind + output folder), so a second queue does not start a twin."""
    if not key:
        return None
    with _REG_LOCK:
        for j in _JOBS.values():
            if j.key == key and j.is_active():
                return j
    return None


def _finish(job: Job, status: str, error: str = "") -> None:
    with job._lock:
        job.status, job.error, job.finished = status, error, time.time()


def start_thread(kind: str, title: str, fn: Callable[[Job], dict | None], key: str | None = None) -> Job:
    """Run ``fn(job)`` in a daemon thread. Its return value (a dict) becomes ``job.result``."""
    running = active_with_key(key or "")
    if running is not None:
        return running
    job = Job(kind, title, key)
    _register(job)

    def body():
        job.status = "running"
        try:
            job.result = fn(job) or {}
            _finish(job, "cancelled" if job.cancelled else "done")
        except JobCancelled as exc:
            job.log(str(exc))
            _finish(job, "cancelled")
        except Exception as exc:  # noqa: BLE001 - the job reports it; the server keeps running
            job.log(traceback.format_exc())
            _finish(job, "error", f"{type(exc).__name__}: {exc}")
            LOG.warning("HDR job %s failed: %s", job.id, exc)

    job._thread = threading.Thread(target=body, name=f"mmx-hdr-{job.id}", daemon=True)
    job._thread.start()
    return job


def kill_tree(proc: subprocess.Popen) -> None:
    """Stop a subprocess and its children (the trainer starts dataloader workers)."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=30)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:  # noqa: BLE001
        pass
    try:
        proc.wait(timeout=10)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def _split_stream(stream, on_line: Callable[[str], None]) -> None:
    """Read text until EOF and call ``on_line`` for every line ended by \\n or \\r (tqdm redraws)."""
    buf = ""
    while True:
        chunk = stream.read(1)
        if not chunk:
            break
        if chunk in ("\n", "\r"):
            if buf:
                on_line(buf)
            buf = ""
        else:
            buf += chunk
    if buf:
        on_line(buf)


def start_process(kind: str, title: str, argv: list[str], *, cwd: str | None = None, env: dict | None = None,
                  log_path: str | None = None, parse: Callable[[Job, str], None] | None = None,
                  key: str | None = None) -> Job:
    """Start ``argv`` in the background; its output goes to the job log (and ``log_path``), line by line to
    ``parse``. Exit code 0 = done, anything else = error (or cancelled when Cancel was pressed)."""
    running = active_with_key(key or "")
    if running is not None:
        return running
    job = Job(kind, title, key)
    job.log_path = log_path
    _register(job)
    kw: dict = {}
    if os.name == "nt":
        kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kw["start_new_session"] = True
    job.proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace", bufsize=1,
                                **kw)
    job.status = "running"

    def body():
        fh = open(log_path, "a", encoding="utf-8") if log_path else None
        try:
            def on_line(line: str):
                job.log(line)
                if fh is not None:
                    fh.write(line + "\n")
                    fh.flush()
                if parse is not None:
                    try:
                        parse(job, line)
                    except Exception:  # noqa: BLE001 - a parser bug must not stop the log
                        pass
            _split_stream(job.proc.stdout, on_line)
            code = job.proc.wait()
            job.returncode = code
            if job.cancelled:
                _finish(job, "cancelled")
            elif code == 0:
                _finish(job, "done")
            else:
                _finish(job, "error", f"the process exited with code {code} (see the log)")
        except Exception as exc:  # noqa: BLE001
            job.log(traceback.format_exc())
            _finish(job, "error", f"{type(exc).__name__}: {exc}")
        finally:
            if fh is not None:
                fh.close()

    job._thread = threading.Thread(target=body, name=f"mmx-hdr-{job.id}", daemon=True)
    job._thread.start()
    return job


def wait_for(job: Job, *, progress: Callable[[int, int], None] | None = None,
             interrupted: Callable[[], bool] | None = None, poll: float = 0.25, timeout: float | None = None) -> Job:
    """Block until the job ends. ``progress(done, total)`` is called when it changes; ``interrupted()`` True ->
    the job is cancelled, then ``JobCancelled`` is raised once it has stopped."""
    last = None
    t0 = time.time()
    while job.is_active():
        if interrupted is not None and interrupted():
            job.cancel()
            job.join(30)
            raise JobCancelled(f"{job.title}: cancelled by the ComfyUI interrupt")
        now = (job.done, job.total)
        if progress is not None and now != last and job.total > 0:
            progress(*now)
            last = now
        if timeout is not None and time.time() - t0 > timeout:
            raise TimeoutError(f"{job.title}: still running after {timeout} s")
        time.sleep(poll)
    job.join(5)
    if progress is not None and job.total > 0:
        progress(job.done, job.total)
    return job


def comfy_waiter():
    """(progress, interrupted) bound to ComfyUI, or (None, None) outside a server (tests)."""
    try:
        import comfy.model_management as mm
        import comfy.utils as cu
    except Exception:  # noqa: BLE001
        return None, None
    holder: dict = {}

    def progress(done: int, total: int):
        bar = holder.get("bar")
        if bar is None or holder.get("total") != total:
            bar = holder["bar"] = cu.ProgressBar(max(1, total))
            holder["total"] = total
        bar.update_absolute(min(done, total), total)

    return progress, mm.processing_interrupted


# ---- persisted "done" state ----------------------------------------------------------------------------------------
class StateFile:
    """``{key: info}`` in a JSON file, written atomically after every item so a crash loses at most one item."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.items: dict = {}
        if self.path.is_file():
            try:
                self.items = json.loads(self.path.read_text(encoding="utf-8")).get("done", {})
            except (OSError, ValueError):
                self.items = {}

    def is_done(self, key: str) -> bool:
        return key in self.items

    def mark(self, key: str, **info) -> None:
        with self._lock:
            self.items[key] = dict(info, at=time.strftime("%Y-%m-%dT%H:%M:%S"))
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".state_", dir=str(self.path.parent))
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"done": self.items}, f, indent=1)
            os.replace(tmp, self.path)

    def forget(self, key: str) -> None:
        with self._lock:
            self.items.pop(key, None)


def folder_signature(folder: str, state_name: str | None = None) -> str:
    """Cheap fingerprint of a work folder (entry count + newest mtime of the top level + the state file), so a node
    re-runs when the folder changed and stays cached when it did not."""
    p = Path(folder)
    if not p.is_dir():
        return "missing"
    n, newest = 0, 0
    try:
        with os.scandir(p) as it:
            for e in it:
                n += 1
                try:
                    newest = max(newest, e.stat().st_mtime_ns)
                except OSError:
                    pass
    except OSError:
        return "unreadable"
    st = ""
    if state_name and (p / state_name).is_file():
        st = str((p / state_name).stat().st_mtime_ns)
    return f"{n}:{newest}:{st}"


# ---- routes ----------------------------------------------------------------------------------------------------------
ROUTES_REGISTERED: list[str] = []


def register_routes() -> bool:
    """Register the job routes once, if the server is up (the Suite's guarded pattern: skip quietly otherwise)."""
    if ROUTES_REGISTERED:
        return True
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:  # noqa: BLE001
        return False
    server = getattr(PromptServer, "instance", None)
    routes = getattr(server, "routes", None) if server is not None else None
    if routes is None or getattr(routes, "get", None) is None:
        return False

    async def list_jobs(request):
        return web.json_response({"jobs": [j.summary() for j in reversed(jobs())]})

    async def job_log(request):
        job = get(request.match_info.get("job_id", ""))
        if job is None:
            return web.json_response({"error": "no such job"}, status=404)
        try:
            n = max(1, min(2000, int(request.query.get("tail", "200"))))
        except ValueError:
            n = 200
        return web.json_response({"job": job.summary(), "lines": job.tail(n), "loss": job.loss[-4000:]})

    async def cancel_job(request):
        job = get(request.match_info.get("job_id", ""))
        if job is None:
            return web.json_response({"error": "no such job"}, status=404)
        job.cancel()
        return web.json_response({"job": job.summary()})

    routes.get("/mmx/hdr/jobs")(list_jobs)
    routes.get("/mmx/hdr/jobs/{job_id}/log")(job_log)
    routes.post("/mmx/hdr/jobs/{job_id}/cancel")(cancel_job)
    ROUTES_REGISTERED.extend(["/mmx/hdr/jobs", "/mmx/hdr/jobs/{job_id}/log", "/mmx/hdr/jobs/{job_id}/cancel"])
    return True


def ui_payload(job: Job) -> dict:
    """What a node returns in ``ui`` so web/hdr_a29_jobs.js can follow the job."""
    return {"mmx_hdr_job": [json.dumps({"job_id": job.id, "kind": job.kind, "status": job.status})]}


def python_exe() -> str:
    return sys.executable
