# Use official Python runtime as base image with CUDA support
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/app/.cache/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Upgrade pip first
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies in stages for better error handling
RUN pip install torch>=2.6.0 torchvision>=0.19.0 torchaudio>=2.6.0 --index-url https://download.pytorch.org/whl/cu124
RUN pip install transformers>=4.45.0 accelerate>=0.33.0 huggingface-hub>=0.25.0
RUN pip install docling>=2.40.0 docling-core>=2.40.0
RUN pip install duckduckgo-search>=6.1.0 requests>=2.31.0 Pillow>=10.0.0 numpy>=1.24.0 pandas>=2.0.0
RUN pip install easyocr>=1.7.0 pypdf>=4.0.0 safetensors>=0.4.0 tokenizers>=0.19.0

# Copy application code
COPY main.py .
COPY process.sh .

# Make scripts executable
RUN chmod +x process.sh

# Create directories for input/output and cache
RUN mkdir -p /app/input /app/output /app/.cache/huggingface

# Expose volume mounts
VOLUME ["/app/input", "/app/output"]

# Set default command
CMD ["./process.sh"]