import runpod


def handler(job):
    """RunPod Hub metadata shim.

    The production worker is copied from the repository-root handler.py by the
    Dockerfile and started through /entrypoint.sh. This lightweight handler is
    present so RunPod Hub can validate the repository structure explicitly.
    """
    return {
        "ok": True,
        "message": "InfiniteTalk RunPod Hub handler detected",
        "operation": job.get("input", {}).get("operation", "health"),
    }


runpod.serverless.start({"handler": handler})
