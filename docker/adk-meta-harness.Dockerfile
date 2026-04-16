ARG TARGETARCH
FROM --platform=linux/${TARGETARCH} python:3.12-slim AS base

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        curl \
        git \
        && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    google-adk>=1.0.0 \
    adk-meta-harness>=0.1.0

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1