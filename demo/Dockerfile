# 1. Start with our verified Python 3.7 image
FROM python:3.7-slim

# 2. Install system dependencies (These rarely change, so they stay at the top)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    python3-dev \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Set up the user and workspace
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app

# --- THE MAGIC CACHE STEP ---
# 4. Copy ONLY the requirements file first
COPY --chown=user requirements.txt .

# 5. Install the heavy packages. 
# Docker will CACHE this layer and only re-run it if requirements.txt changes.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# 6. NOW copy the rest of your app (including app.py and models)
# Since this is at the end, changing app.py won't trigger Step 5 anymore!
COPY --chown=user . /home/user/app

# 7. Environment and Launch settings
EXPOSE 7860
ENV PYTHONUNBUFFERED=1
CMD ["python", "-u", "app.py"]