# PORTED FROM: 1038lab/ComfyUI-Minimax-H3-Promptor :: py/config_manager.py
# Comfyui-Minimax-H3-Promptor — author 1038lab, GPL-3.0

from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path

from . import PROMPTOR_ROOT

DEFAULT_CONFIG = {
    "version": "1.2.0",
    "providers": {
        "prov_1720000001": {
            "name": "openai",
            "type": "openai",
            "api_base": "https://api.openai.com/v1",
            "api_key": "",
            "model": "gpt-5",
            "enabled": True,
            "batch_vision": True,
        },
        "prov_1720000004": {
            "name": "ollama",
            "type": "ollama",
            "api_base": "http://localhost:11434",
            "model": "llama3.2",
            "enabled": True,
            "batch_vision": False,
        },
    },
    "defaults": {
        "vision_provider": "prov_1720000001",
        "vision_model": "gpt-5",
        "vision_temperature": 0.2,
        "vision_max_tokens": 4096,
        "promptor_provider": "prov_1720000001",
        "promptor_model": "gpt-5",
        "promptor_temperature": 0.7,
        "promptor_max_tokens": 4096,
    },
}


class ConfigManager:
    """Manages configuration for the H3 Promptor extension."""

    def __init__(self, root_dir: str | Path | None = None):
        self.root_dir = Path(root_dir) if root_dir else PROMPTOR_ROOT
        self.config_path = self.root_dir / "config.json"
        self.example_path = self.root_dir / "config.example.json"
        self._config: dict | None = None

    def load(self) -> dict:
        if self._config is not None:
            return self._config

        if not self.config_path.exists():
            self._create_default_config()

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._config = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            from .utils import log_error

            log_error(f"Failed to load config.json: {e}. Using defaults.")
            self._config = DEFAULT_CONFIG.copy()

        needs_migration = False
        migrated_providers = {}
        for k, v in self._config.get("providers", {}).items():
            if not k.startswith("prov_"):
                new_key = f"prov_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
                v["name"] = v.get("name", k)
                if "batch_vision" not in v:
                    v["batch_vision"] = v.get("type", "openai").lower() not in [
                        "ollama",
                        "lmstudio",
                        "llamacpp",
                    ]
                migrated_providers[new_key] = v
                defs = self._config.get("defaults", {})
                if defs.get("vision_provider") == k:
                    defs["vision_provider"] = new_key
                if defs.get("promptor_provider") == k:
                    defs["promptor_provider"] = new_key
                needs_migration = True
            else:
                migrated_providers[k] = v

        if needs_migration:
            self._config["providers"] = migrated_providers
            self._config["version"] = "1.2.0"
            self.save(self._config)

        self._config = self._merge_defaults(self._config, DEFAULT_CONFIG, is_root=True)
        return self._config

    def get_provider_config(self, provider_name: str) -> dict:
        config = self.load()
        providers = config.get("providers", {})
        if provider_name.lower() not in providers:
            if provider_name.lower() in DEFAULT_CONFIG.get("providers", {}):
                return DEFAULT_CONFIG["providers"][provider_name.lower()]
            from .utils import log_error

            log_error(
                f"Provider '{provider_name}' not found in config. "
                f"Available: {list(providers.keys())}"
            )
            return {}
        return providers[provider_name.lower()]

    def find_provider_by_display_name(self, display_name: str) -> str:
        import re

        config = self.load()
        providers = config.get("providers", {})
        if not display_name:
            return config.get("defaults", {}).get("vision_provider", "")

        if display_name in providers:
            return display_name

        clean_target = re.sub(r"\s*\([^)]*\)\s*$", "", display_name).strip().lower()
        norm_target = re.sub(r"[\s_\-]+", "-", clean_target)

        for k, v in providers.items():
            name = v.get("name", k)
            model_name = v.get("model", "")
            clean_raw = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()
            norm_raw = re.sub(r"[\s_\-]+", "-", clean_raw.lower())
            expected = f"{clean_raw} ({model_name})" if model_name else clean_raw
            if display_name == expected or display_name.lower() == expected.lower():
                return k
            if display_name.lower() == name.lower() or display_name.lower() == k.lower():
                return k
            if clean_target and (clean_target == clean_raw.lower() or clean_target == name.lower()):
                return k
            if norm_target and (norm_target == norm_raw or norm_target in norm_raw):
                return k
            if model_name and (
                clean_target in model_name.lower()
                or norm_target in model_name.lower().replace("_", "-")
            ):
                return k

        tokens = [t for t in re.split(r"[^a-zA-Z0-9\.]+", clean_target) if len(t) > 1]
        if tokens:
            for k, v in providers.items():
                name = v.get("name", k).lower()
                model_name = v.get("model", "").lower()
                combined = f"{name} {model_name}"
                if all(t in combined for t in tokens):
                    return k

        for k in providers:
            if k.lower() == display_name.lower():
                return k

        return display_name

    def get_defaults(self) -> dict:
        config = self.load()
        return config.get("defaults", DEFAULT_CONFIG["defaults"])

    def get_available_providers(self) -> list[str]:
        config = self.load()
        return list(config.get("providers", {}).keys())

    def _create_default_config(self):
        from .utils import log_info

        if self.example_path.exists():
            shutil.copy2(self.example_path, self.config_path)
            log_info("Created config.json from config.example.json")
        else:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)
            log_info("Created config.json with default settings")
        log_info(f"Please edit {self.config_path} to set your API keys.")

    @staticmethod
    def _merge_defaults(user_config: dict, defaults: dict, is_root: bool = True) -> dict:
        merged = defaults.copy()
        if is_root and "providers" in user_config and len(user_config["providers"]) > 0:
            merged["providers"] = {}
        for key, value in user_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigManager._merge_defaults(value, merged[key], is_root=False)
            else:
                merged[key] = value
        return merged

    def reload(self):
        self._config = None
        return self.load()

    def save(self, new_config: dict) -> bool:
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(new_config, f, indent=2, ensure_ascii=False)
            self._config = new_config
            return True
        except Exception as e:
            from .utils import log_error

            log_error(f"Failed to save config.json: {e}")
            return False


_config_manager: ConfigManager | None = None


def get_config_manager() -> ConfigManager:
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(root_dir=PROMPTOR_ROOT)
    return _config_manager
