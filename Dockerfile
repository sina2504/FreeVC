# ✅ Use official PyTorch + CUDA 12.1 runtime
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy project files
COPY . .

# Install system-level dependencies (incl. gcc for building webrtcvad)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Launch the app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]