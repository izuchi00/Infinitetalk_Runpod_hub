# RunPod Hub detection shim.
# The actual serverless worker implementation lives at repository root: /handler.py
# Importing it executes runpod.serverless.start({"handler": handler}).

from handler import *  # noqa: F401,F403
