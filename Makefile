# Nairobi Flood Guard AI - common tasks.
#
#   make demo            one-command local demo: asset check, cache warm-up,
#                        then API + Streamlit together (Ctrl-C stops both)
#   make app             Streamlit UI only (http://localhost:8501)
#   make api             FastAPI only (http://localhost:8000/docs)
#   make test            unit tests (synthetic fixtures; no 100MB graph needed)
#   make lint            ruff
#   make refresh-cache   fetch live rainfall, rescore wards, rerun rerouting,
#                        evaluate threshold-crossing alerts
#   make verify-assets   fail loudly if a data asset is an LFS pointer
#   make docker          docker compose up --build (API + app containers)

PY ?= python3

.PHONY: demo app api test lint refresh-cache verify-assets docker

verify-assets:
	$(PY) -m scripts.verify_data_assets

# Streamlit's first-run email prompt reads /dev/tty (not stdin) and blocks
# headless startup; an empty-credentials file is the documented bypass.
STREAMLIT_CREDS := $(HOME)/.streamlit/credentials.toml

$(STREAMLIT_CREDS):
	@mkdir -p $(HOME)/.streamlit
	@printf '[general]\nemail = ""\n' > $@

app: $(STREAMLIT_CREDS)
	streamlit run app.py

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000

test:
	$(PY) -m pytest tests -v

lint:
	ruff check .

refresh-cache: verify-assets
	$(PY) -m scripts.refresh_cache

# Warm the rerouting cache so the demo never pays the graph-load cost on
# stage, then serve both processes from one terminal. Requires a POSIX shell.
demo: verify-assets $(STREAMLIT_CREDS)
	@echo "==> Warming rerouting cache (skips network rainfall fetch)"
	$(PY) -m scripts.refresh_cache --skip-rainfall
	@echo ""
	@echo "==> Starting services"
	@echo "    App:      http://localhost:8501"
	@echo "    API:      http://localhost:8000  (docs at /docs)"
	@echo "    GTFS-RT:  http://localhost:8000/reroutes/gtfs-rt"
	@echo "    Ctrl-C stops both."
	@uvicorn api.main:app --host 0.0.0.0 --port 8000 & \
	API_PID=$$!; \
	trap 'kill $$API_PID 2>/dev/null' EXIT; \
	streamlit run app.py

docker:
	docker compose up --build
