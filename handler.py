import base64
import binascii
import json
import logging
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import librosa
import runpod
import websocket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-media-infinitetalk")

SERVER_ADDRESS = os.getenv("SERVER_ADDRESS", "127.0.0.1")
COMFY_HTTP = f"http://{SERVER_ADDRESS}:8188"
COMFY_WS = f"ws://{SERVER_ADDRESS}:8188/ws"

WORKFLOWS = {
    "single": Path("/I2V_single.json"),
    "multi": Path("/I2V_multi.json"),
}


def _now():
    return time.perf_counter()


def _truncate(value, length=80):
    if not value:
        return value
    text = str(value)
    return text if len(text) <= length else text[:length] + "..."


def _download(url: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return destination


def _base64_to_file(data: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.write_bytes(base64.b64decode(data, validate=True))
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Invalid base64 input: {exc}") from exc
    return destination


def _resolve_input(job_input, prefix: str, task_dir: Path, filename: str, required=True):
    path_key = f"{prefix}_path"
    url_key = f"{prefix}_url"
    base64_key = f"{prefix}_base64"

    if job_input.get(path_key):
        path = Path(str(job_input[path_key])).expanduser()
        if not path.exists():
            raise ValueError(f"{path_key} does not exist: {path}")
        return path.resolve()

    destination = task_dir / filename
    if job_input.get(url_key):
        return _download(str(job_input[url_key]), destination)
    if job_input.get(base64_key):
        return _base64_to_file(str(job_input[base64_key]), destination)

    if required:
        raise ValueError(f"One of {path_key}, {url_key}, or {base64_key} is required")
    return None


def _duration(path: Path) -> float:
    return float(librosa.get_duration(path=str(path)))


def _align_4k1(frames: int) -> int:
    frames = max(1, int(frames))
    if frames == 1:
        return 1
    return ((frames - 1 + 3) // 4) * 4 + 1


def _resolve_max_frame(value, audio1: Path, audio2: Path | None, multi_audio_type: str, fps=25):
    if value not in (None, "", "auto"):
        return max(1, int(value))

    d1 = _duration(audio1)
    if audio2 is None:
        seconds = d1
    else:
        d2 = _duration(audio2)
        seconds = max(d1, d2) if multi_audio_type == "para" else d1 + d2
    return _align_4k1(round(seconds * fps))


def _load_workflow(person_count: str):
    path = WORKFLOWS[person_count]
    return json.loads(path.read_text(encoding="utf-8"))


def _configure_workflow(
    workflow: dict,
    *,
    person_count: str,
    image_path: Path,
    audio1_path: Path,
    audio2_path: Path | None,
    prompt: str,
    width: int,
    height: int,
    max_frame: int,
    steps: int,
    force_offload: bool,
    seed: int,
    cfg: float,
    shift: float,
    multi_audio_type: str,
):
    workflow["284"]["inputs"]["image"] = str(image_path)
    workflow["125"]["inputs"]["audio"] = str(audio1_path)
    workflow["241"]["inputs"]["positive_prompt"] = prompt
    workflow["245"]["inputs"]["value"] = width
    workflow["246"]["inputs"]["value"] = height
    workflow["270"]["inputs"]["value"] = max_frame
    workflow["194"]["inputs"]["multi_audio_type"] = multi_audio_type

    sampler = workflow["128"]["inputs"]
    sampler["steps"] = steps
    sampler["force_offload"] = force_offload
    sampler["seed"] = seed
    sampler["cfg"] = cfg
    sampler["shift"] = shift

    if "192" in workflow:
        workflow["192"]["inputs"]["force_offload"] = force_offload

    if person_count == "multi":
        if audio2_path is None:
            raise ValueError("Multi-speaker mode requires a second audio input")
        workflow["307"]["inputs"]["audio"] = str(audio2_path)

    return workflow


def _queue_prompt(prompt: dict, client_id: str):
    data = json.dumps({"prompt": prompt, "client_id": client_id}).encode("utf-8")
    request = urllib.request.Request(f"{COMFY_HTTP}/prompt", data=data)
    request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def _get_history(prompt_id: str):
    with urllib.request.urlopen(f"{COMFY_HTTP}/history/{prompt_id}", timeout=30) as response:
        return json.loads(response.read())


def _run_workflow(prompt: dict):
    client_id = str(uuid.uuid4())
    ws = websocket.create_connection(f"{COMFY_WS}?clientId={client_id}", timeout=30)
    workflow_started = _now()
    node_timings = []
    current_node = None
    current_started = None

    try:
        prompt_id = _queue_prompt(prompt, client_id)["prompt_id"]
        logger.info("ComfyUI prompt queued: %s", prompt_id)

        while True:
            message = ws.recv()
            if not isinstance(message, str):
                continue
            payload = json.loads(message)
            if payload.get("type") != "executing":
                continue

            data = payload.get("data", {})
            if data.get("prompt_id") != prompt_id:
                continue

            node = data.get("node")
            now = _now()

            if current_node is not None and current_started is not None:
                node_timings.append({
                    "node": str(current_node),
                    "seconds": round(now - current_started, 3),
                })

            if node is None:
                break

            current_node = node
            current_started = now
            logger.info("Executing node: %s", node)

        history = _get_history(prompt_id).get(prompt_id, {})
        outputs = history.get("outputs", {})
        video_paths = []
        for node_id, node_output in outputs.items():
            for item in node_output.get("gifs", []):
                fullpath = item.get("fullpath")
                if fullpath and os.path.isfile(fullpath):
                    video_paths.append(Path(fullpath))

        if not video_paths:
            raise RuntimeError("InfiniteTalk completed but no output MP4 was found")

        video_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return video_paths[0], round(_now() - workflow_started, 3), node_timings
    finally:
        try:
            ws.close()
        except Exception:
            pass


def handler(job: dict):
    job_input = job.get("input") or {}
    operation = str(job_input.get("operation", "talking_avatar")).strip().lower()

    if operation == "health":
        return {
            "ok": True,
            "worker": "ai-media-platform-infinitetalk",
            "supports": ["single", "multi"],
            "model_delivery": "runpod_cached_huggingface_bundle",
            "wav2vec_prebaked": False,
            "wav2vec_cached_bundle": True,
            "keep_wav2vec_gpu": os.getenv("INFINITALK_KEEP_WAV2VEC_GPU", "1") == "1",
        }

    if operation != "talking_avatar":
        return {"ok": False, "error": f"Unsupported operation: {operation}"}

    total_started = _now()
    stage_timings = {}
    task_dir = Path(f"/tmp/ai-media-{uuid.uuid4().hex}")
    task_dir.mkdir(parents=True, exist_ok=True)

    try:
        person_count = str(job_input.get("person_count", "single")).strip().lower()
        if person_count not in WORKFLOWS:
            raise ValueError("person_count must be 'single' or 'multi'")

        multi_audio_type = str(job_input.get("multi_audio_type", "para")).strip().lower()
        if multi_audio_type not in {"para", "add"}:
            raise ValueError("multi_audio_type must be 'para' or 'add'")

        width = int(job_input.get("width", 512))
        height = int(job_input.get("height", 512))
        if width < 256 or height < 256 or width > 1024 or height > 1024:
            raise ValueError("width and height must be between 256 and 1024")
        if width % 16 or height % 16:
            raise ValueError("width and height must be divisible by 16")

        steps = int(job_input.get("steps", 3))
        if not 2 <= steps <= 12:
            raise ValueError("steps must be between 2 and 12")

        force_offload = bool(job_input.get("force_offload", False))
        seed = int(job_input.get("seed", 2))
        cfg = float(job_input.get("cfg", 1.0))
        shift = float(job_input.get("shift", 11.0))
        prompt_text = str(job_input.get(
            "prompt",
            "A professional podcast speaker talking naturally to camera, realistic facial movement, natural blinking, subtle head movement, accurate lip synchronization, stable identity.",
        ))

        log_input = {
            "person_count": person_count,
            "multi_audio_type": multi_audio_type,
            "width": width,
            "height": height,
            "steps": steps,
            "force_offload": force_offload,
            "max_frame": job_input.get("max_frame", 151),
            "image_url": _truncate(job_input.get("image_url")),
            "wav_url": _truncate(job_input.get("wav_url")),
            "wav_url_2": _truncate(job_input.get("wav_url_2")),
        }
        logger.info("Received job config: %s", log_input)

        runpod.serverless.progress_update(job, "Preparing InfiniteTalk inputs")
        t = _now()
        image_path = _resolve_input(job_input, "image", task_dir, "input_image.jpg")
        audio1_path = _resolve_input(job_input, "wav", task_dir, "speaker1.wav")
        audio2_path = _resolve_input(
            job_input,
            "wav_2",
            task_dir,
            "speaker2.wav",
            required=(person_count == "multi"),
        )
        stage_timings["input_prepare_seconds"] = round(_now() - t, 3)

        max_frame_value = job_input.get("max_frame", 151)
        max_frame = _resolve_max_frame(
            max_frame_value,
            audio1_path,
            audio2_path,
            multi_audio_type,
        )

        t = _now()
        workflow = _configure_workflow(
            _load_workflow(person_count),
            person_count=person_count,
            image_path=image_path,
            audio1_path=audio1_path,
            audio2_path=audio2_path,
            prompt=prompt_text,
            width=width,
            height=height,
            max_frame=max_frame,
            steps=steps,
            force_offload=force_offload,
            seed=seed,
            cfg=cfg,
            shift=shift,
            multi_audio_type=multi_audio_type,
        )
        stage_timings["workflow_config_seconds"] = round(_now() - t, 3)

        runpod.serverless.progress_update(
            job,
            f"Generating InfiniteTalk ({person_count}, {width}x{height}, {steps} steps)",
        )
        output_path, workflow_seconds, node_timings = _run_workflow(workflow)
        stage_timings["comfy_workflow_seconds"] = workflow_seconds

        runpod.serverless.progress_update(job, "Encoding benchmark result")
        t = _now()
        output_bytes = output_path.read_bytes()
        encoded_video = base64.b64encode(output_bytes).decode("ascii")
        stage_timings["output_encode_seconds"] = round(_now() - t, 3)

        total_seconds = round(_now() - total_started, 3)
        return {
            "ok": True,
            "job_id": str(job.get("id", "local")),
            "provider": "runpod-serverless",
            "model": "infinitetalk",
            "person_count": person_count,
            "multi_audio_type": multi_audio_type,
            "width": width,
            "height": height,
            "max_frame": max_frame,
            "steps": steps,
            "force_offload": force_offload,
            "seed": seed,
            "cfg": cfg,
            "shift": shift,
            "stage_timings": stage_timings,
            "node_timings": node_timings,
            "total_seconds": total_seconds,
            "video_base64": encoded_video,
            "content_type": "video/mp4",
            "output_filename": output_path.name,
        }

    except Exception as exc:
        logger.exception("InfiniteTalk job failed")
        return {
            "ok": False,
            "error": str(exc),
            "total_seconds": round(_now() - total_started, 3),
            "stage_timings": stage_timings,
        }
    finally:
        shutil.rmtree(task_dir, ignore_errors=True)


runpod.serverless.start({"handler": handler})
