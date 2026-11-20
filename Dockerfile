# syntax=docker/dockerfile:1

# Build native libraries and the Python environment.
FROM ubuntu:24.04 AS builder

# Keep the package manager version fixed independently of the base image.
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /uvx /bin/

ARG TARGETARCH
ARG OPENSLIDE_VERSION=4.0.0
ARG VIPS_VERSION=8.16.0

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:/opt/native/bin:$PATH \
    LD_LIBRARY_PATH=/opt/native/lib \
    PKG_CONFIG_PATH=/opt/native/lib/pkgconfig \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy

RUN test "$TARGETARCH" = amd64 || (echo 'This image requires --platform linux/amd64' >&2; exit 1)

# Install native build dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libgl1 \
    libgomp1 \
    meson \
    ninja-build \
    pkg-config \
    python3 \
    python3-dev \
    python3-venv \
    libarchive-dev \
    libcairo2-dev \
    libexif-dev \
    libexpat1-dev \
    libffi-dev \
    libfftw3-dev \
    libglib2.0-dev \
    libjpeg-turbo8-dev \
    liblcms2-dev \
    libopenjp2-7-dev \
    liborc-0.4-dev \
    libpng-dev \
    libsqlite3-dev \
    libtiff-dev \
    libwebp-dev \
    libxml2-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install the additional image library development headers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdk-pixbuf-2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Build OpenSlide and libvips.
WORKDIR /tmp/native

RUN curl -fL --retry 3 -o openslide.tar.xz \
        "https://github.com/openslide/openslide/releases/download/v${OPENSLIDE_VERSION}/openslide-${OPENSLIDE_VERSION}.tar.xz" \
    && tar -xf openslide.tar.xz \
    && meson setup "openslide-${OPENSLIDE_VERSION}/build" "openslide-${OPENSLIDE_VERSION}" \
        --prefix=/opt/native --libdir=lib --buildtype=release -Ddoc=disabled -Dtest=disabled \
    && meson compile -C "openslide-${OPENSLIDE_VERSION}/build" \
    && meson install -C "openslide-${OPENSLIDE_VERSION}/build"

RUN curl -fL --retry 3 -o vips.tar.xz \
        "https://github.com/libvips/libvips/releases/download/v${VIPS_VERSION}/vips-${VIPS_VERSION}.tar.xz" \
    && tar -xf vips.tar.xz \
    && meson setup "vips-${VIPS_VERSION}/build" "vips-${VIPS_VERSION}" \
        --prefix=/opt/native --libdir=lib --buildtype=release \
        -Dintrospection=disabled -Dexamples=false -Dcplusplus=false \
        -Dopenslide=enabled -Dopenslide-module=disabled \
        -Dtiff=enabled -Djpeg=enabled -Dpng=enabled -Dopenjpeg=enabled \
    && meson compile -C "vips-${VIPS_VERSION}/build" \
    && meson test -C "vips-${VIPS_VERSION}/build" --print-errorlogs \
    && meson install -C "vips-${VIPS_VERSION}/build"

# Finalize the native runtime with the narrow libtiff compatibility backport.
# Keep the unmodified native/dependency build layers reusable when editing it.
COPY docker/patch_openslide.py /tmp/patch_openslide.py
RUN python3 /tmp/patch_openslide.py \
        "/tmp/native/openslide-${OPENSLIDE_VERSION}/src/openslide-decode-tiff.c" \
    && meson compile -C "/tmp/native/openslide-${OPENSLIDE_VERSION}/build" \
    && meson install -C "/tmp/native/openslide-${OPENSLIDE_VERSION}/build"

# Use an independently managed Python; Ubuntu's Python remains available to Meson.
RUN uv python install 3.10 && uv venv /opt/venv --python 3.10
WORKDIR /opt/project
COPY pyproject.toml README.md ./
COPY histomil ./histomil
COPY scripts ./scripts
# Keep the CUDA wheels fixed while resolving the Poetry project and test tools.
RUN --mount=type=cache,target=/root/.cache/uv \
    printf 'torch==2.10.0+cu128\ntorchvision==0.25.0+cu128\n' > /tmp/constraints.txt \
    && uv pip install --python /opt/venv/bin/python --index-url https://download.pytorch.org/whl/cu128 \
        'torch==2.10.0+cu128' 'torchvision==0.25.0+cu128' \
    && uv pip install --python /opt/venv/bin/python -c /tmp/constraints.txt . \
    && uv pip install --python /opt/venv/bin/python -c /tmp/constraints.txt 'pytest>=7,<8' \
    && uv pip check --python /opt/venv/bin/python \
    && uv pip freeze --python /opt/venv/bin/python > /opt/venv/requirements-resolved.txt
COPY tests ./tests
RUN python -m pytest tests/test_metrics.py tests/test_models.py tests/test_weighted_sampler.py tests/test_convert_data_splits.py

FROM ubuntu:24.04 AS runtime
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /uvx /bin/
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/venv/bin:/opt/native/bin:$PATH \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    LD_LIBRARY_PATH=/opt/native/lib \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    XDG_CACHE_HOME=/cache \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    TRIDENT_HOME=/cache/trident \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    openssh-server \
    rsync \
    tini \
    tmux \
    python3 \
    python3-venv \
    libarchive13t64 \
    libcairo2 \
    libexif12 \
    libexpat1 \
    libffi8 \
    libfftw3-double3 \
    libgdk-pixbuf-2.0-0 \
    libgl1 \
    libglib2.0-0t64 \
    libgomp1 \
    libjpeg-turbo8 \
    liblcms2-2 \
    libopenjp2-7 \
    liborc-0.4-0t64 \
    libpng16-16t64 \
    libsqlite3-0 \
    libtiff6 \
    libwebp7 \
    libwebpdemux2 \
    libwebpmux3 \
    libxml2 \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/native /opt/native
COPY --from=builder /opt/python /opt/python
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/project/pyproject.toml /opt/project/README.md /opt/project/
COPY --from=builder /opt/project/histomil /opt/project/histomil
COPY --from=builder /opt/project/scripts /opt/project/scripts
COPY docker/validate.py /opt/container/validate.py
COPY docker/profile.sh /etc/profile.d/histomil.sh
RUN mkdir -p /cache /workspace \
    && uv pip check --python /opt/venv/bin/python \
    && python /opt/container/validate.py \
    && histomil-splits --help \
    && histomil-train --help \
    && histomil-grid --help \
    && histomil-predict --help \
    && histomil-heatmap --help
WORKDIR /workspace
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
