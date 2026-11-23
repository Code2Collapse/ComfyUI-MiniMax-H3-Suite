# Node panels on a cached re-run (L9.290)

## What was measured (core 0.39.0, frontend 1.53.10)

- An unchanged re-queue is served from the output cache. `ExecutionList.add_node` (comfy_execution/graph.py) walks
  only into inputs that are **not** cached, so a cached node that is not an output node is never visited and
  `execute()` never sends its `executed` message. Output nodes are always visited, and on a cache hit they get their
  stored ui back (`emit_cached_output`).
- `execution_cached` lists every cached node id at the start of the run. Cached nodes get no `executing`.
- Frontend 1.53.10 keeps each workflow tab's last ui payloads (`changeTracker.nodeOutputs` / `restoreOutputs`) and
  re-shows images from them. It never calls `onExecuted` again, so custom panels start empty after a reload, a tab
  switch or a history load. Before this change, queueing again did not fill them either.
- Registry (202 nodes, measured in a fresh interpreter): 44 non-output nodes return ui. 6 of them have a picture panel
  (Track Crop, Mask Prep, Drift QC, Subject Crop / Advanced, Context Windows, Tone Compensate). The rest show a
  text report or a preview image. The 18 output nodes that return ui were already resent by core.

## Design

**Core's own flag, not output nodes and not a second store.** Since 0.19, core has had a schema flag
`has_intermediate_output`. At the end of a run that finishes, core resends the cached ui of every such node it did not
execute, through the same `executed` message a fresh run sends. Core reads the flag in only two places, that sweep and
`/object_info`, so it adds **no execution**: an unwired node still never runs, and a cached node is not re-run. Every
consumer (the six panels, the text reports, the frontend's own image previews, `/history`) gets exactly what a fresh
run gives it, with no per-panel code.

- All 44 nodes declare `has_intermediate_output=True`. Ids, inputs, outputs, defaults and widget layout are
  unchanged, so saved workflows load and run as before.
- **No processing node becomes an output node.** An output node always runs, even with nothing wired downstream.
  That would force sampling, VAE or RAFT work on unwired nodes.
- The kit status strip (web/kit.js) reads `execution_cached`. A cached node shows "cached". When core's resend
  arrives it shows "cached · last result". If the run stops before the resend, it shows "cached · result not shown
  (run stopped)", and the hover says why and what to do.
- Guard: tests/test_cached_ui_resend.py runs core's real `PromptExecutor` on the pack. It also fails when a new
  non-output node returns ui without the flag, or when the flag sits on a node with no ui.

## Rejected

- *Frontend store keyed by node id + input hash, plus a summary in `node.properties`.* Core already resends the
  payload whenever the node is in a finished run, and the frontend already keeps it per tab. A third copy would put
  run data (boxes, prompts, plans) into saved and shared workflow files. It could also show a stale result that no
  longer matches the inputs, and preview images in it would point at temp files that are deleted when the server
  restarts.

## Limits (explained in the UI where they show)

1. Core resends only at the end of a run that **finishes**. A run that stops with an error or a cancel does not refill a
   cached node's panel (the strip says so). Queue again after fixing it: nothing re-runs, and the result comes back.
2. After a page reload or a tab switch, a panel shows its empty state ("Run the node to see ...") until the next queue.
   That queue re-runs nothing that is cached.
3. A node the RAM-pressure cache dropped is simply re-run (core 0.39 default cache, D0.40). That is correct, just
   not free.
4. Panels inside subgraphs are refilled the same way (core addresses them as `subgraph:node`).
