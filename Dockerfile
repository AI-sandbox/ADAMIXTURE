FROM python:3.12-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

ARG VERSION=""
RUN if [ -n "${VERSION}" ]; then \
        export SETUPTOOLS_SCM_PRETEND_VERSION="${VERSION}"; \
    elif [ ! -d ".git" ]; then \
        export SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0"; \
    fi && \
    CFLAGS="-fopenmp" \
    LDFLAGS="-fopenmp -Wl,--no-as-needed -lopenblas" \
    pip wheel --no-cache-dir --wheel-dir /wheels .


FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/AI-sandbox/ADAMIXTURE" \
      org.opencontainers.image.description="ADAMIXTURE: Fast Biobank-Scale Population Genetics Clustering (CPU)"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libopenblas0-pthread \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /tmp/matplotlib /data \
    && chmod 1777 /tmp/matplotlib

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

RUN python -c "import adamixture; from adamixture.src.utils_c import em, sqp, tools, bvls, br_qn, snp_reader, clumppling" && \
    adamixture --help >/dev/null && \
    adamixture-project --help >/dev/null

WORKDIR /data
CMD ["adamixture", "--help"]

