"""Local Hugging Face transformers provider: any repo id or local folder (ours).

Not limited to one family: whatever AutoProcessor / AutoModelForImageTextToText / AutoModelForMultimodalLM /
AutoModelForCausalLM can load (Qwen3-VL, Gemma 4, Qwopus, SmolVLM, text-only chat models ...). What the model can
take is read from its processor: an image processor means pictures and video sheets, an audio feature extractor means
audio. A repo that is not on disk is downloaded file by file into <models>/LLM/<org>--<name>/ only when the run may
download (downloads.py). trust_remote_code is off unless the user turns it on.
"""

from __future__ import annotations

import fnmatch
import gc
import io
import os
import re
import time
from typing import Any, Callable

from . import downloads as D
from .modes import WriterError
from .text import strip_thinking

_CACHE: dict[tuple, tuple[Any, Any, bool]] = {}      # key -> (model, processor-or-tokenizer, is_processor)
REPO_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
BASE_PATTERNS = ("*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "tokenizer*", "*.tiktoken")
DTYPES = ("auto", "bfloat16", "float16", "float32")
DEVICES = ("auto", "cuda", "cpu")
QUANTIZATIONS = ("none", "8-bit (bitsandbytes)", "4-bit (bitsandbytes)")


def _import(name: str):
    try:
        if name == "torch":
            import torch
            return torch
        if name == "transformers":
            import transformers
            return transformers
        if name == "huggingface_hub":
            import huggingface_hub
            return huggingface_hub
    except Exception as e:  # noqa: BLE001
        raise WriterError(f"{name} could not be imported ({type(e).__name__}: {e}); the local Hugging Face provider "
                          f"needs it.") from None
    raise ValueError(name)


def resolve_model(model_id: str, preset: dict | None, root: str) -> tuple[str, str | None, str | None]:
    """(local folder, repo id or None, revision or None). The folder may not exist yet."""
    mid = (model_id or "").strip().strip('"')
    revision = None
    if not mid and preset:
        mid, revision = preset["repo"], preset.get("revision")
    if not mid:
        raise WriterError("Name the model: type a Hugging Face repo id or a local folder in model_id, or pick a preset.")
    if os.path.isabs(mid):
        if not os.path.isdir(mid):
            raise WriterError(f"The folder {mid} does not exist.")
        return mid, None, None
    local = os.path.join(root, mid)
    if os.path.isdir(local) and os.path.isfile(os.path.join(local, "config.json")):
        return local, None, None
    if "@" in mid:                                     # org/name@revision
        mid, revision = mid.split("@", 1)
    if not REPO_ID.match(mid):
        raise WriterError(f"{mid!r} is neither a Hugging Face repo id (org/name) nor a folder in {root}.")
    return os.path.join(root, mid.replace("/", "--")), mid, revision


PARTIAL_MARK = ".mmx_download_incomplete"


def _present(folder: str) -> bool:
    """A model folder is usable when it has config.json and no unfinished download of ours."""
    return (os.path.isfile(os.path.join(folder, "config.json"))
            and not os.path.exists(os.path.join(folder, PARTIAL_MARK)))


def download(repo: str, revision: str | None, target: str, trust_remote_code: bool,
             progress: Callable[[str], None], check_interrupt: Callable[[], None] | None = None) -> None:
    """File by file; files already complete are skipped by huggingface_hub, so a stopped download resumes."""
    hub = _import("huggingface_hub")
    patterns = BASE_PATTERNS + (("*.py",) if trust_remote_code else ())
    try:
        info = hub.HfApi().model_info(repo, revision=revision, files_metadata=True)
    except Exception as e:  # noqa: BLE001
        raise WriterError(f"Cannot read {repo} on Hugging Face ({type(e).__name__}: {e}). Check the repo id, your "
                          f"network, or log in with huggingface-cli for a gated model.") from None
    files = [s for s in (info.siblings or []) if any(fnmatch.fnmatch(os.path.basename(s.rfilename), p)
                                                    for p in patterns)]
    if not any(s.rfilename.endswith(".safetensors") for s in files):
        raise WriterError(f"{repo} has no .safetensors weights; this provider loads safetensors only.")
    total = sum(int(getattr(s, "size", 0) or 0) for s in files)
    progress(f"downloading {repo} ({len(files)} files, {total / 2**30:.2f} GB) to {target}")
    t0 = time.perf_counter()
    os.makedirs(target, exist_ok=True)
    mark = os.path.join(target, PARTIAL_MARK)
    with open(mark, "w", encoding="utf-8") as fh:
        fh.write(f"{repo}@{info.sha or revision}\n")
    for i, s in enumerate(files, 1):
        if check_interrupt is not None:
            check_interrupt()
        progress(f"download {i}/{len(files)}: {s.rfilename} ({int(getattr(s, 'size', 0) or 0) / 2**20:.1f} MB)")
        try:
            hub.hf_hub_download(repo, s.rfilename, revision=info.sha or revision, local_dir=target)
        except Exception as e:  # noqa: BLE001
            raise WriterError(f"Download of {s.rfilename} from {repo} failed ({type(e).__name__}: {e}); run the node "
                              f"again to resume.") from None
    os.remove(mark)
    progress(f"downloaded {repo} in {time.perf_counter() - t0:.1f} s")


def _interrupt():
    try:
        import comfy.model_management as mm
        return mm.throw_exception_if_processing_interrupted
    except Exception:  # noqa: BLE001
        return None


class _Interrupt:
    def __init__(self, check: Callable[[], None] | None):
        self.check = check

    def __call__(self, input_ids, scores, **kwargs):
        if self.check is not None:
            self.check()
        return False


class HFProvider:
    def __init__(self, settings, progress: Callable[..., None]):
        self.settings = settings
        self.progress = progress
        root = D.llm_dir()
        folder, repo, revision = resolve_model(settings.model_id, settings.preset, root)
        if not _present(folder):
            if repo is None and os.path.exists(os.path.join(folder, PARTIAL_MARK)):
                raise WriterError(f"{folder} holds an unfinished download; run with its repo id to resume it.")
            if repo is None:
                raise WriterError(f"{folder} has no config.json: it is not a transformers model folder.")
            if not D.downloads_allowed(settings.downloads):
                raise WriterError(f"{repo} is not in {root} and downloads are off ({D.why_not(settings.downloads)}). "
                                  f"Download it to {folder}, or allow downloads.")
            download(repo, revision, folder, bool(settings.trust_remote_code), progress, _interrupt())
        self.folder = folder
        self.model = repo or os.path.basename(folder)
        self.name = f"huggingface transformers ({self.model})"
        self._load()

    def _key(self) -> tuple:
        s = self.settings
        return (os.path.normcase(os.path.abspath(self.folder)), s.device, s.dtype, s.quantization,
                bool(s.trust_remote_code))

    def _load(self) -> None:
        key = self._key()
        if key in _CACHE:
            self.lm, self.proc, self.is_processor = _CACHE[key]
            self.progress(f"model: {self.folder} (already loaded)")
            return
        torch = _import("torch")
        tf = _import("transformers")
        s = self.settings
        trc = bool(s.trust_remote_code)
        device = s.device if s.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            raise WriterError("device is 'cuda' but no CUDA GPU is available; pick 'auto' or 'cpu'.")
        kwargs: dict[str, Any] = {"trust_remote_code": trc}
        if s.dtype != "auto":
            kwargs["dtype"] = getattr(torch, s.dtype)
        elif device == "cpu":
            kwargs["dtype"] = torch.float32
        else:
            kwargs["dtype"] = "auto"
        if s.quantization != "none":
            try:
                import bitsandbytes  # noqa: F401
            except Exception:  # noqa: BLE001
                raise WriterError("quantization needs the bitsandbytes package, which is not installed; pick "
                                  "'none' or install bitsandbytes into ComfyUI's Python.") from None
            if device != "cuda":
                raise WriterError("bitsandbytes quantization needs a CUDA GPU; pick quantization 'none' on CPU.")
            kwargs["quantization_config"] = tf.BitsAndBytesConfig(load_in_8bit=s.quantization.startswith("8"),
                                                                  load_in_4bit=s.quantization.startswith("4"))
            kwargs["device_map"] = "auto"
        t0 = time.perf_counter()
        proc, is_processor = None, False
        try:
            proc = tf.AutoProcessor.from_pretrained(self.folder, trust_remote_code=trc)
            is_processor = not isinstance(proc, tf.PreTrainedTokenizerBase)
        except Exception:  # noqa: BLE001 - text-only repos ship a tokenizer, no processor
            proc = None
        if proc is None:
            try:
                proc = tf.AutoTokenizer.from_pretrained(self.folder, trust_remote_code=trc)
            except Exception as e:  # noqa: BLE001
                raise WriterError(f"Cannot load a processor or tokenizer from {self.folder} ({type(e).__name__}: {e}).") from None
        lm, errors = None, []
        for cls_name in ("AutoModelForImageTextToText", "AutoModelForMultimodalLM", "AutoModelForCausalLM"):
            cls = getattr(tf, cls_name, None)
            if cls is None:
                continue
            try:
                lm = cls.from_pretrained(self.folder, **kwargs)
                break
            except TypeError as e:
                if "dtype" in str(e) and "dtype" in kwargs:          # older transformers: torch_dtype
                    kw = dict(kwargs)
                    kw["torch_dtype"] = kw.pop("dtype")
                    try:
                        lm = cls.from_pretrained(self.folder, **kw)
                        break
                    except Exception as e2:  # noqa: BLE001
                        errors.append(f"{cls_name}: {type(e2).__name__}: {e2}")
                else:
                    errors.append(f"{cls_name}: {type(e).__name__}: {e}")
            except Exception as e:  # noqa: BLE001
                errors.append(f"{cls_name}: {type(e).__name__}: {str(e)[:200]}")
        if lm is None:
            hint = "" if trc else " If the model card says it needs custom code, turn on trust_remote_code."
            raise WriterError(f"transformers could not load {self.model}: " + "; ".join(errors[-2:]) + hint)
        if "device_map" not in kwargs:
            lm = lm.to(device)
        lm.eval()
        self.lm, self.proc, self.is_processor = lm, proc, is_processor
        _CACHE[key] = (lm, proc, is_processor)
        self.progress(f"model: loaded {self.folder} on {device} in {time.perf_counter() - t0:.1f} s")

    def capabilities(self) -> dict[str, bool]:
        if not self.is_processor:
            return {"images": False, "audio": False}
        p = self.proc
        return {"images": getattr(p, "image_processor", None) is not None,
                "audio": (getattr(p, "feature_extractor", None) is not None
                          or getattr(p, "audio_processor", None) is not None)}

    def _conversation(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from PIL import Image
        convo = []
        for m in messages:
            content: list[dict[str, Any]] = []
            for p in m["parts"]:
                if p["type"] == "text":
                    content.append({"type": "text", "text": p["text"]})
                elif p["type"] == "image":
                    content.append({"type": "image", "image": Image.open(io.BytesIO(p["data"])).convert("RGB")})
                elif p["type"] == "audio":
                    content.append({"type": "audio", "audio": _wav_to_array(p["data"])})
            if not self.is_processor:
                content = "\n\n".join(c["text"] for c in content if c["type"] == "text")  # type: ignore[assignment]
            convo.append({"role": m["role"], "content": content})
        return convo

    def _encode(self, convo: list[dict[str, Any]]):
        kw = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        try:
            return self.proc.apply_chat_template(convo, **kw)
        except Exception as first:  # noqa: BLE001 - some templates reject a system role: fold it into the first user turn
            if convo and convo[0]["role"] == "system" and len(convo) > 1:
                sys_msg, rest = convo[0], [dict(m) for m in convo[1:]]
                if isinstance(rest[0]["content"], str):
                    rest[0]["content"] = f"{sys_msg['content']}\n\n{rest[0]['content']}"
                else:
                    rest[0]["content"] = list(sys_msg["content"]) + list(rest[0]["content"])
                try:
                    return self.proc.apply_chat_template(rest, **kw)
                except Exception:  # noqa: BLE001
                    pass
            raise WriterError(f"{self.model} has no usable chat template ({type(first).__name__}: {first}).") from None

    def complete(self, messages: list[dict[str, Any]], temperature: float = 1.0, top_p: float = 0.95,
                 max_tokens: int = 2048, seed: int | None = None,
                 check_interrupt: Callable[[], None] | None = None) -> dict[str, Any]:
        torch = _import("torch")
        tf = _import("transformers")
        inputs = self._encode(self._conversation(messages))
        device = next(self.lm.parameters()).device
        inputs = inputs.to(device) if hasattr(inputs, "to") else {k: v.to(device) for k, v in inputs.items()}
        n_in = int(inputs["input_ids"].shape[-1])
        if seed is not None:
            torch.manual_seed(int(seed))
        gen: dict[str, Any] = {"max_new_tokens": int(max_tokens),
                               "stopping_criteria": tf.StoppingCriteriaList([_Interrupt(check_interrupt)])}
        if temperature and temperature > 0:
            gen.update(do_sample=True, temperature=float(temperature), top_p=float(top_p))
        else:
            gen.update(do_sample=False)
        with torch.inference_mode():
            out = self.lm.generate(**inputs, **gen)
        new = out[0, n_in:]
        tok = getattr(self.proc, "tokenizer", self.proc)
        text = tok.decode(new, skip_special_tokens=True)
        return {"text": strip_thinking(text), "finish_reason": "length" if int(new.shape[-1]) >= int(max_tokens) else "stop",
                "usage": {"prompt_tokens": n_in, "completion_tokens": int(new.shape[-1])}}

    def close(self, unload: bool = True) -> None:
        if not unload:
            return
        _CACHE.pop(self._key(), None)
        self.lm = self.proc = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        try:
            import comfy.model_management as mm
            mm.soft_empty_cache()
        except Exception:  # noqa: BLE001
            pass


def _wav_to_array(data: bytes):
    import wave

    import numpy as np
    with wave.open(io.BytesIO(data), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
