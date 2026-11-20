# EgoCut - reproducible GPU image.  Build:  docker build -t egocut .
# Run one video:  docker run --gpus all -v /path/to/videos:/data -v /path/to/out:/out egocut /data/P01_11.MP4 /out/P01_11.json
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /egocut
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN bash weights/download_weights.sh
ENTRYPOINT ["bash", "/egocut/run.sh"]
