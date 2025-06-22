FROM python:3.12-slim

WORKDIR /app

# Copy requirements file first (for better caching)
COPY requirements_gpu.txt .

# Install necessary dependencies, including FFmpeg, Calibre, and OpenGL libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    ffmpeg \
    curl \
    libegl1 \
    libopengl0 \
    libxcb-cursor0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install uv

# Install pip 24.0 first
RUN uv pip install --system --no-cache-dir --upgrade pip==24.0

# Install Calibre
RUN curl -sS https://download.calibre-ebook.com/linux-installer.sh | sh /dev/stdin

# ✅ Make Calibre CLI tools globally accessible
RUN ln -s /opt/calibre/ebook-convert /usr/local/bin/ebook-convert && \
    ln -s /opt/calibre/ebook-meta /usr/local/bin/ebook-meta

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Install Python dependencies (after upgrading pip)
RUN uv pip install --system --no-cache-dir -r requirements_gpu.txt

# Fix six package version
RUN uv pip install --system --no-cache-dir --upgrade six==1.17.0

# Copy the rest of the application files
COPY . .

# ✅ Create the `generated_audiobooks` directory
RUN mkdir -p /app/generated_audiobooks

# Expose the port
EXPOSE 7860

# Run the application with Uvicorn
CMD ["uvicorn", "--access-log", "app:app", "--host", "0.0.0.0", "--port", "7860"]
