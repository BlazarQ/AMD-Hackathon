# 1. Use a lightweight, stable Python base image
FROM python:3.10-slim

# 2. Set environment variables to keep logs clean and reduce memory bloat
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# 3. Set the working directory inside the container
WORKDIR /app

# 4. THE C++ TRAP: Install compilers before touching Python requirements
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/* # 5. Copy requirements and install Python packages
# We force the CPU build arguments to ensure maximum compatibility with the grading server
COPY requirements.txt .
ENV CMAKE_ARGS="-DGGML_CPU=ON"
RUN pip install -r requirements.txt

# 6. Copy the fully integrated Python script
COPY agent.py .

# 7. Copy the local 2GB model weights directly into the container
COPY models/Llama-3.2-3B-Instruct-Q4_K_M.gguf /models/Llama-3.2-3B-Instruct-Q4_K_M.gguf

# 8. Create the mounting directories the auto-grader expects
RUN mkdir -p /input /output

# 9. Define the execution command
CMD ["python", "agent.py"]