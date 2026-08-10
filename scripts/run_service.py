"""Container entrypoint: select which service to run.

    python -m scripts.run_service api   # FastAPI on 0.0.0.0:8000
    python -m scripts.run_service app   # Streamlit on 0.0.0.0:8501

Honours the conventional PORT env var (Render/Heroku style) when set.
"""

from __future__ import annotations

import os
import sys

import uvicorn


def main() -> None:
    service = sys.argv[1] if len(sys.argv) > 1 else "api"
    port = int(os.environ.get("PORT", "8000" if service == "api" else "8501"))

    if service == "api":
        uvicorn.run("api.main:app", host="0.0.0.0", port=port)
    elif service == "app":
        from streamlit.web import cli as stcli

        sys.argv = [
            "streamlit",
            "run",
            "app.py",
            "--server.port",
            str(port),
            "--server.address",
            "0.0.0.0",
            "--server.headless",
            "true",
        ]
        raise SystemExit(stcli.main())
    else:
        raise SystemExit(f"Unknown service {service!r}; expected 'api' or 'app'")


if __name__ == "__main__":
    main()
