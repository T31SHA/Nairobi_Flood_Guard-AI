# Nairobi Flood Guard AI - single image serving both the API and the app.
#
#   docker build -t nairobi-flood-guard .
#   docker run -p 8000:8000 nairobi-flood-guard            # API
#   docker run -p 8501:8501 nairobi-flood-guard app        # Streamlit UI
#
# or simply `docker compose up` (see docker-compose.yml).
#
# The ~100 MB road network and other data assets are baked into the image,
# so the build context must be a full clone with Git LFS objects pulled
# (`git lfs pull`), not a ZIP download - scripts/verify_data_assets.py runs
# during the build and fails loudly on an LFS pointer file.

FROM python:3.12-slim

WORKDIR /srv/app

# Install dependencies first so code edits don't bust the package layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Fail the build here, not on stage, if a data asset is an LFS pointer.
RUN python -m scripts.verify_data_assets

# Skip Streamlit's first-run email prompt (it blocks headless startup).
RUN mkdir -p /root/.streamlit && printf '[general]\nemail = ""\n' > /root/.streamlit/credentials.toml

ENV PYTHONUNBUFFERED=1

EXPOSE 8000 8501

# Default service: the API. Pass "app" (or any streamlit/uvicorn command)
# to run the dashboard instead - see docker-compose.yml.
ENTRYPOINT ["python", "-m", "scripts.run_service"]
CMD ["api"]
