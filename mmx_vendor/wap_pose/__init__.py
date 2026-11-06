"""WanAnimatePreprocessV2's ONNX pose stack (YOLO person detector + ViTPose wholebody), carried so N8 StrongestPose
runs without that pack installed (ledger L9.73 / L7.20 D4).

models/onnx_models.py and pose_utils/pose2d_utils.py are byte-identical copies of ComfyUI-WanAnimatePreprocessV2's
files (Apache-2.0; Alibaba Wan team, kijai, Code2Collapse). Never edit them here: change WanAnimatePreprocessV2, then
run tools/sync_vendored.py --write in the workspace. These __init__ files are this pack's own. See ../VENDORED.md.
"""
