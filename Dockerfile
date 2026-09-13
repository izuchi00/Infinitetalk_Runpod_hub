# syntax=docker/dockerfile:1.7

FROM wlsdml1114/engui_genai-base_blackwell:1.1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    INFINITALK_KEEP_WAV2VEC_GPU=1

WORKDIR /

RUN apt-get update && apt-get install -y --no-install-recommends \
      wget curl git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/worker-requirements.txt

RUN python -m pip install --upgrade pip && \
    python -m pip install -r /tmp/worker-requirements.txt && \
    rm -f /tmp/worker-requirements.txt

RUN git clone https://github.com/comfyanonymous/ComfyUI.git /ComfyUI && \
    cd /ComfyUI && \
    git checkout 8ccc0c94 && \
    python -m pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Comfy-Org/ComfyUI-Manager.git && \
    cd ComfyUI-Manager && \
    python -m pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/city96/ComfyUI-GGUF.git && \
    cd ComfyUI-GGUF && \
    python -m pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-KJNodes.git && \
    cd ComfyUI-KJNodes && \
    python -m pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    cd ComfyUI-VideoHelperSuite && \
    python -m pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/orssorbit/ComfyUI-wanBlockswap.git

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-MelBandRoFormer.git && \
    cd ComfyUI-MelBandRoFormer && \
    python -m pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git && \
    cd ComfyUI-WanVideoWrapper && \
    git checkout 088128b224242e110d3906c6750e9a3a348a659b && \
    python -m pip install -r requirements.txt

# Model weights are supplied at runtime by RunPod's cached Hugging Face bundle.
RUN mkdir -p \
    /ComfyUI/models/diffusion_models \
    /ComfyUI/models/loras \
    /ComfyUI/models/vae \
    /ComfyUI/models/text_encoders \
    /ComfyUI/models/clip_vision \
    /ComfyUI/models/transformers

COPY patches/patch_wav2vec_residency.py /tmp/patch_wav2vec_residency.py
RUN python /tmp/patch_wav2vec_residency.py && \
    rm -f /tmp/patch_wav2vec_residency.py

COPY I2V_single.json /I2V_single.json
COPY I2V_multi.json /I2V_multi.json
COPY handler.py /handler.py
COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
