# Build context is the REPO ROOT, not this directory — see docker-compose.yml
# (`context: .`, `dockerfile: webapp/backend/Dockerfile`). The backend imports
# `from src.utils.geometry import ...`, which lives outside webapp/backend/,
# so a context scoped to just this folder would fail at import time with
# "ModuleNotFoundError: No module named 'src'".
FROM python:3.11-slim-bookworm

# There is no standalone "kicad-cli" apt package — kicad-cli ships inside the
# main `kicad` package. Debian bookworm's default repo only has KiCad 6.0.11,
# which predates kicad-cli (added in KiCad 7). Pulling from
# bookworm-backports gets 9.0.x, which has it. (Verified against
# packages.debian.org; not build-tested in this environment — if the
# backports version has moved on, adjust the pinned suite/version below.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && echo "deb http://deb.debian.org/debian bookworm-backports main" \
        > /etc/apt/sources.list.d/backports.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends -t bookworm-backports kicad \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY webapp/backend/requirements.txt webapp/backend/requirements.txt
RUN pip install --no-cache-dir -r webapp/backend/requirements.txt

COPY src/ src/
COPY webapp/__init__.py webapp/__init__.py
COPY webapp/backend/ webapp/backend/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "webapp.backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
