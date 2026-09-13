from pathlib import Path

path = Path('/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper/multitalk/nodes.py')
text = path.read_text(encoding='utf-8')
old = '            wav2vec2.to(offload_device)\n'
new = '''            # AI Media Platform Phase 11: optionally keep Wav2Vec resident on GPU
            # across jobs on a warm worker. This avoids repeated CPU<->GPU transfers.
            import os as _os
            if _os.getenv("INFINITALK_KEEP_WAV2VEC_GPU", "1") != "1":
                wav2vec2.to(offload_device)
'''
if old not in text:
    raise RuntimeError('Expected Wav2Vec offload line was not found; wrapper revision changed.')
path.write_text(text.replace(old, new, 1), encoding='utf-8')
print('Patched Wav2Vec residency behavior.')
