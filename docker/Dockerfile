# Dockerfile for Autonomous Data Analyst Agent sandbox execution
# Phase 2: Isolated container environment for untrusted Python analysis code.

FROM python:3.11-slim

# Prevent python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create non-root user 'sandbox' with UID 1000
RUN groupadd -g 1000 sandbox && \
    useradd -u 1000 -g sandbox -m -s /bin/bash sandbox

# Install required numerical, data science, and plotting packages
RUN pip install --no-cache-dir \
    pandas==2.2.3 \
    numpy==2.1.3 \
    scipy==1.14.1 \
    scikit-learn==1.5.2 \
    matplotlib==3.9.2 \
    seaborn==0.13.2

# Setup workspace directories
RUN mkdir -p /workspace/data /workspace/script /workspace/output && \
    chown -R sandbox:sandbox /workspace

# Switch to non-root user
USER sandbox

# Working directory where output charts and reports will be written
WORKDIR /workspace/output

# Default command runs the injected analysis script
CMD ["python", "/workspace/script/analysis_script.py"]

