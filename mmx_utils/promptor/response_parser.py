# PORTED FROM: 1038lab/ComfyUI-Minimax-H3-Promptor :: py/response_parser.py
# Comfyui-Minimax-H3-Promptor — author 1038lab, GPL-3.0

from __future__ import annotations

import json
import re


class ResponseParser:
    """Dedicated utility class for processing LLM Vision output into clean dictionaries."""

    @staticmethod
    def extract_json_content(content: str) -> dict | None:
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        brace_match = re.search(r"\{.*\}", content, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        return None

    @staticmethod
    def parse_vision_response(content: str, target_keys: list[str]) -> dict:
        if not content or not content.strip():
            return {k: "LLM returned empty response." for k in target_keys}

        parsed = ResponseParser.extract_json_content(content)
        return_dict: dict[str, str] = {}

        if parsed and isinstance(parsed, dict):
            if all(k in parsed for k in target_keys):
                return {k: str(parsed[k]).strip() for k in target_keys}

            if len(target_keys) == 1:
                target_key = target_keys[0]
                raw_key = target_key.replace("<", "").replace(">", "").strip()
                if target_key in parsed:
                    return {target_key: str(parsed[target_key]).strip()}
                if raw_key in parsed:
                    return {target_key: str(parsed[raw_key]).strip()}
                for pk, pv in parsed.items():
                    if raw_key.lower() in pk.lower() or pk.lower() in raw_key.lower():
                        return {target_key: str(pv).strip()}
                for generic_k in ["description", "desc", "analysis", "caption", "detail", "content", "summary", "result", "text"]:
                    for pk, pv in parsed.items():
                        if generic_k in pk.lower() and str(pv).strip():
                            return {target_key: str(pv).strip()}
                for pv in parsed.values():
                    if isinstance(pv, str) and pv.strip():
                        return {target_key: pv.strip()}
                return {target_key: json.dumps(parsed, ensure_ascii=False)}

            unmatched_keys = []
            used_parsed_keys = set()
            for target_key in target_keys:
                raw_key = target_key.replace("<", "").replace(">", "").strip()
                matched = False
                if target_key in parsed:
                    return_dict[target_key] = str(parsed[target_key]).strip()
                    used_parsed_keys.add(target_key)
                    matched = True
                elif raw_key in parsed:
                    return_dict[target_key] = str(parsed[raw_key]).strip()
                    used_parsed_keys.add(raw_key)
                    matched = True
                else:
                    for pk, pv in parsed.items():
                        if pk not in used_parsed_keys and raw_key.lower() in pk.lower():
                            return_dict[target_key] = str(pv).strip()
                            used_parsed_keys.add(pk)
                            matched = True
                            break
                if not matched:
                    unmatched_keys.append(target_key)

            if unmatched_keys:
                remaining_parsed = [
                    v for k, v in parsed.items() if k not in used_parsed_keys and str(v).strip()
                ]
                for i, target_key in enumerate(unmatched_keys):
                    if i < len(remaining_parsed):
                        return_dict[target_key] = str(remaining_parsed[i]).strip()
                    else:
                        return_dict[target_key] = "LLM analyzed this item."
            return return_dict

        if len(target_keys) == 1:
            return {target_keys[0]: content.strip()}

        sections = {}
        matches = list(
            re.finditer(
                r"(?:<Picture\s*(\d+)>|Picture\s*(\d+)[:\-]?|Image\s*(\d+)[:\-]?|\[Picture\s*(\d+)\]|^\s*(\d+)\.\s+)",
                content,
                re.IGNORECASE | re.MULTILINE,
            )
        )
        if matches:
            for idx, m in enumerate(matches):
                num = next((int(g) for g in m.groups() if g is not None), idx + 1)
                key = f"<Picture {num}>"
                start_pos = m.end()
                end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
                body = content[start_pos:end_pos].strip()
                if body.startswith(":") or body.startswith("-"):
                    body = body[1:].strip()
                sections[key] = body

        for i, target_key in enumerate(target_keys):
            if target_key in sections and sections[target_key]:
                return_dict[target_key] = sections[target_key]
            elif f"<Picture {i + 1}>" in sections and sections[f"<Picture {i + 1}>"]:
                return_dict[target_key] = sections[f"<Picture {i + 1}>"]

        missing = [k for k in target_keys if k not in return_dict]
        if missing:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
            for i, target_key in enumerate(missing):
                if i < len(paragraphs):
                    return_dict[target_key] = paragraphs[i]
                else:
                    return_dict[target_key] = content.strip()

        return return_dict
