"""H3 Prompt Writer as a workflow node (owner order 2026-11-23, design docs/PLAN_h3_prompt_writer_node.md).

Prompt-writing logic ported from third_party/ComfyUI-MiniMaxH3-Prompt-Writer (MIT, H3 Prompt Writer contributors):
the mode wrappers, the reference policy, the official-format audit and the repair pass. The media store, the HTTP
routes and the floating window are not ported: the node takes IMAGE / VIDEO / AUDIO sockets instead.

Every module here imports only the standard library at module level. torch, numpy, PIL, transformers,
huggingface_hub and llama_cpp are imported inside the functions that need them, so a missing optional library fails
one provider with a sentence and never the pack.
"""
