FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

# Copy custom SSL certificates if present (for corporate VPN/proxy)
COPY .devcontainer/certs /tmp/custom-certs/
RUN if [ -n "$(ls -A /tmp/custom-certs 2>/dev/null)" ]; then \
        cp /tmp/custom-certs/*.crt /usr/local/share/ca-certificates/ 2>/dev/null || true; \
        update-ca-certificates; \
    fi && rm -rf /tmp/custom-certs

# Install Python build dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    git

# Install Python 3.12 (stable version to avoid SSL issues with 3.14)
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install uv (Python package manager) system-wide
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv* /usr/local/bin/ \
    && chmod +x /usr/local/bin/uv*

# Clean up
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*
