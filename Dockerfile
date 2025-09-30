# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies for scientific Python and cartopy (GEOS/PROJ) and build tools
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       g++ \
       gcc \
       pkg-config \
       libgeos-dev \
       libproj-dev \
       proj-bin \
       proj-data \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY req.txt ./req.txt
RUN pip install --no-cache-dir -r req.txt

# Copy project files
COPY . .

# Default command to run the script
CMD ["python", "Arome_Test.py"]


