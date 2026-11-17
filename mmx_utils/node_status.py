"""Per-node-type status for the front-end layout engine (A28, ORDERS 29; web/mmx_node_layout.js).

PORTED FROM: ComfyUI-CustomNodePacks nodes/_node_status.py (Code2Collapse, Apache-2.0; same author).

An input declared with ``extra_dict={"c2c_show_if_status": "<key>"}`` is shown only while the node-status route answers
``{"status": {"<key>": true}}`` for its node type. A node module registers a provider with ``register_status``; the
provider must only LOOK (file checks through the node's own resolvers): it never downloads, never loads a model and
never walks a model tree, because the front end asks on page load and after every run.

Route: ``GET /mmx/node_status?type=<NodeType>``. CNP serves ``/c2c/node_status`` for its own providers; the two paths
never collide, so both packs load together in any order. The Suite's engine copy asks ``/c2c/node_status`` first and
``/mmx/node_status`` on a 404 (docs/audit/A28/MMX_ENGINE_PORT.md in the C2C workspace). No Suite node declares
``c2c_show_if_status`` yet; the route exists so the Suite works alone once one does.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict

LOG = logging.getLogger("MiniMaxH3.NodeStatus")

ROUTE = "/mmx/node_status"

_PROVIDERS: Dict[str, Callable[[], dict]] = {}
_REGISTERED_ON: set[int] = set()


def register_status(node_type: str, provider: Callable[[], dict]) -> None:
    _PROVIDERS[str(node_type)] = provider


def status_of(node_type: str) -> dict:
    """The provider's answer; KeyError for a type without one."""
    out = _PROVIDERS[str(node_type)]()
    return {str(k): v for k, v in dict(out).items()}


def register_routes(server) -> bool:
    """Add ``GET /mmx/node_status`` to ``server.routes`` once per server. Returns True when it was added now."""
    if server is None or id(server) in _REGISTERED_ON:
        return False
    from aiohttp import web

    @server.routes.get(ROUTE)
    async def _mmx_node_status(request: "web.Request") -> "web.Response":
        node_type = request.rel_url.query.get("type", "")
        if node_type not in _PROVIDERS:
            return web.json_response({"error": f"no status for node type {node_type!r}"}, status=404)
        try:
            return web.json_response({"type": node_type, "status": status_of(node_type)})
        except Exception as exc:  # noqa: BLE001 - the front end then shows the control (honest default)
            LOG.warning("node status %s failed: %s", node_type, exc)
            return web.json_response({"error": str(exc)}, status=500)

    _REGISTERED_ON.add(id(server))
    return True
