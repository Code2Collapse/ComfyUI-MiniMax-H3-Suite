"""Where local prompt models live, whether this run may download one, and progress text (ours).

Downloads happen only inside a node run, never at import or test time, and only when all of these allow it:
the node's ``downloads`` choice, the C2C setting "Never download models" (``c2c.models.neverDownload``, read from the
saved settings so it works with or without CustomNodePacks; ``C2C_NO_DOWNLOADS=1`` wins) and the Prompt Writer's
offline switch ``MMX_PROMPT_WRITER_OFFLINE``.
"""

from __future__ import annotations

import os

from .modes import WriterError

DOWNLOAD_CHOICES = ("download if missing", "never download")
NEVER_DOWNLOAD_SETTING = "c2c.models.neverDownload"


def never_download() -> bool:
    try:
        from mmx_utils.llm_http import offline
        if offline():
            return True
    except Exception:  # noqa: BLE001
        pass
    env = os.environ.get("C2C_NO_DOWNLOADS")
    if env is not None and env != "":
        return env == "1"
    try:
        from mmx_utils import _c2c_saved_setting
        return _c2c_saved_setting.get(NEVER_DOWNLOAD_SETTING, False) is True
    except Exception:  # noqa: BLE001 - no settings module: nothing saved
        return False


def downloads_allowed(choice: str) -> bool:
    return choice == DOWNLOAD_CHOICES[0] and not never_download()


def why_not(choice: str) -> str:
    if choice != DOWNLOAD_CHOICES[0]:
        return "downloads is set to 'never download'"
    if os.environ.get("C2C_NO_DOWNLOADS") == "1":
        return "C2C_NO_DOWNLOADS=1 is set"
    return "the C2C setting 'Never download models' or MMX_PROMPT_WRITER_OFFLINE is on"


def llm_dir() -> str:
    """ComfyUI's LLM model folder: a registered 'LLM' / 'llm' folder path, else <models>/LLM."""
    try:
        import folder_paths  # type: ignore
    except Exception:  # noqa: BLE001
        raise WriterError("Local models need ComfyUI's folder_paths (run inside ComfyUI).") from None
    for key in ("LLM", "llm"):
        try:
            paths = folder_paths.get_folder_paths(key)
        except Exception:  # noqa: BLE001 - not registered
            paths = []
        if paths:
            return paths[0]
    return os.path.join(folder_paths.models_dir, "LLM")


def guide_dir() -> str | None:
    try:
        from .guides import GUIDE_SUBDIR
        return os.path.join(llm_dir(), GUIDE_SUBDIR)
    except WriterError:
        return None


class Progress:
    """Progress text on the node (core's send_progress_text when available) and in the report."""

    def __init__(self, node_id: str | None, report: list[str]):
        self.node_id = node_id
        self.report = report

    def __call__(self, text: str, to_report: bool = True) -> None:
        if to_report:
            self.report.append(text)
        if not self.node_id:
            return
        try:
            from server import PromptServer  # type: ignore
            inst = getattr(PromptServer, "instance", None)
            if inst is not None and hasattr(inst, "send_progress_text"):
                inst.send_progress_text(text, self.node_id)
        except Exception:  # noqa: BLE001 - progress text is a courtesy
            pass
