"""Fail loudly instead of silently proceeding with a corrupt or placeholder
asset (e.g. an un-pulled Git LFS pointer file).

Run: python -m scripts.verify_data_assets

The exact failure mode this guards against: the repo's ~100 MB road network
is tracked with Git LFS, so a ZIP download or a clone without LFS gets a
~134-byte pointer file in its place - and every graph-dependent feature then
fails (or silently falls back) much later and much more confusingly.
"""

from __future__ import annotations

from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# Real-file minimums, set comfortably below actual size but well above an LFS
# pointer (which is typically well under 200 bytes).
CHECKS = {
    "Data/nairobi_road_network.graphml": 50_000_000,  # real file is ~101.5 MB
    "Data/floods.gpkg": 10_000_000,  # real file is ~33 MB
    "Models/flood_model.joblib": 1_000_000,
}


def check_assets(base: Path = BASE) -> list[str]:
    """Return a list of human-readable problems (empty when all is well)."""
    problems = []
    for rel_path, min_bytes in CHECKS.items():
        p = base / rel_path
        if not p.exists():
            problems.append(f"MISSING: {rel_path}")
            continue
        size = p.stat().st_size
        if size < min_bytes:
            head = p.open("rb").read(200)
            hint = (
                " -> this is a Git LFS pointer, not the real file. "
                "Run `git lfs install && git lfs pull`, or see "
                "scripts/rebuild_road_network.py for a regeneration fallback."
                if b"git-lfs" in head
                else ""
            )
            problems.append(
                f"TOO SMALL: {rel_path} is {size}B, expected >={min_bytes}B{hint}"
            )
    return problems


def main() -> None:
    problems = check_assets()
    if problems:
        raise SystemExit(
            "Data asset check FAILED:\n" + "\n".join(f"  - {p}" for p in problems)
        )
    print("All critical data assets present and correctly sized.")


if __name__ == "__main__":
    main()
