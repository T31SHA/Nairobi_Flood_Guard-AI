"""Out-of-time sanity check: March 2026 Nairobi floods vs model predictions.

This is a coarse, retrospective check using secondary reporting — NOT a
rigorous relabel or equivalent to the registry's spatial-CV metrics. The
model was trained on April 2024 labels with historical rainfall features;
predictions here use those same static features.

Sources for reported-affected areas (every entry is cited inline below):
  - Build brief anchor neighborhoods (Mathare, Eastleigh, Mukuru Kwa Reuben)
  - Kenya Interior Ministry 37-area situational list (March 2026, via ReliefWeb)
  - Ushahidi Nairobi Floods situational tracking
  - News coverage aggregated on Wikipedia (March 2026 Kenya floods)

Run: python -m scripts.validate_against_2026_event
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import joblib
import pandas as pd

from Utils.feature_engineering import engineer_features

BASE = Path(__file__).resolve().parent.parent
REGISTRY_PATH = BASE / "Models" / "model_registry.json"
FLOODS_GPKG = BASE / "Data" / "floods.gpkg"


@dataclass(frozen=True)
class GroundTruthEntry:
  neighborhood: str
  ward: str | None
  reported_2026: bool
  source: str
  notes: str = ""


# Ward names must match Data/floods.gpkg exactly where mapped.
# Neighborhoods without a ward in our layer are listed with ward=None.
GROUND_TRUTH: tuple[GroundTruthEntry, ...] = (
  GroundTruthEntry(
    "Mathare",
    "Mathare North Ward",
    True,
    "Build brief §3 #10; Ushahidi Nairobi Floods situational reports",
  ),
  GroundTruthEntry(
    "Eastleigh",
    "Eastleigh North Ward",
    True,
    "Build brief §3 #10; multiple news reports (Mombasa Rd / Eastleigh corridor)",
  ),
  GroundTruthEntry(
    "Eastleigh South",
    "Eastleigh South Ward",
    True,
    "Interior Ministry 37-area list (March 2026, ReliefWeb situational update)",
  ),
  GroundTruthEntry(
    "Mukuru Kwa Reuben",
    "Kwa Reuben Ward",
    True,
    "Build brief §3 #10; Ushahidi / community reporting",
  ),
  GroundTruthEntry(
    "Korogocho",
    "Korogocho Ward",
    True,
    "Interior Ministry list; river-corridor overlap with UNOSAT 2024 mapping",
  ),
  GroundTruthEntry(
    "Dandora",
    "Dandora Area I Ward",
    True,
    "Interior Ministry list; UNOSAT 2024 river-corridor overlap",
  ),
  GroundTruthEntry(
    "Kariobangi",
    "Kariobangi North Ward",
    True,
    "Interior Ministry list",
  ),
  GroundTruthEntry(
    "Kayole",
    "Kayole Central Ward",
    True,
    "Interior Ministry list; news reports",
  ),
  GroundTruthEntry(
    "Ruai",
    "Ruai Ward",
    True,
    "News reports (impassable Mombasa Rd corridor); Interior Ministry list",
  ),
  GroundTruthEntry(
    "Parklands",
    "Parklands/highridge Ward",
    True,
    "Interior Ministry 37-area list",
  ),
  GroundTruthEntry(
    "Kilimani",
    "Kilimani Ward",
    True,
    "Interior Ministry list",
  ),
  GroundTruthEntry(
    "Kibera",
    None,
    True,
    "News / situational reports — no 'Kibera' ward in floods.gpkg ward layer",
    notes="Not mappable to ward layer",
  ),
  GroundTruthEntry(
    "South B",
    "South C Ward",
    False,
    "Control: reported-flooded anchor neighborhoods only; South B not in primary lists",
    notes="Proxy ward for South B area",
  ),
)


def load_predictions() -> pd.DataFrame:
  registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
  wards = engineer_features(gpd.read_file(FLOODS_GPKG))
  model = joblib.load(BASE / registry["model_path"])
  feature_cols = registry["feature_cols"]
  X = wards[feature_cols].fillna(wards[feature_cols].median())
  wards["flood_prob"] = model.predict_proba(X)[:, 1]
  wards["threshold"] = registry["threshold"]
  return wards, registry


def risk_tier(prob: float, threshold: float) -> str:
  if prob >= 0.70:
    return "Critical"
  if prob >= 0.45:
    return "High"
  if prob >= threshold:
    return "Moderate"
  return "Low"


def main() -> None:
  wards, registry = load_predictions()
  threshold = registry["threshold"]
  nairobi = wards[wards["county"].str.lower() == "nairobi"]

  rows = []
  for entry in GROUND_TRUTH:
    if entry.ward is None:
      rows.append(
        {
          "neighborhood": entry.neighborhood,
          "ward": "(not in ward layer)",
          "predicted_tier": "—",
          "flood_prob": None,
          "above_threshold": None,
          "reported_flooded_2026": "Y" if entry.reported_2026 else "N",
          "source": entry.source,
          "notes": entry.notes or "ward not in dataset",
        }
      )
      continue
    match = nairobi[nairobi["ward"] == entry.ward]
    if match.empty:
      prob = None
      tier = "—"
      above = None
      note = f"ward '{entry.ward}' not found in Nairobi layer"
    else:
      prob = float(match.iloc[0]["flood_prob"])
      tier = risk_tier(prob, threshold)
      above = prob >= threshold
      note = entry.notes
    rows.append(
      {
        "neighborhood": entry.neighborhood,
        "ward": entry.ward,
        "predicted_tier": tier,
        "flood_prob": prob,
        "above_threshold": above,
        "reported_flooded_2026": "Y" if entry.reported_2026 else "N",
        "source": entry.source,
        "notes": note,
      }
    )

  df = pd.DataFrame(rows)
  print("=" * 72)
  print("OUT-OF-TIME VALIDATION: March 2026 Nairobi floods")
  print("=" * 72)
  print(
    "\nDISCLAIMER: Retrospective sanity check using secondary reporting.\n"
    "NOT equivalent to spatial-CV metrics in model_registry.json.\n"
    "Model uses April 2024 training labels + static historical rainfall features.\n"
  )
  print(f"Registry threshold: {threshold:.3f}")
  print()
  display = df.copy()
  display["flood_prob"] = display["flood_prob"].map(
    lambda x: f"{x:.1%}" if x is not None else "—"
  )
  display["above_threshold"] = display["above_threshold"].map(
    lambda x: "Y" if x is True else ("N" if x is False else "—")
  )
  print(display.to_string(index=False))
  print()

  mappable = df[df["flood_prob"].notna() & df["reported_flooded_2026"].eq("Y")]
  if len(mappable):
    hits = int(mappable["above_threshold"].sum())
    total = len(mappable)
    print(
      f"Mappable reported-flooded neighborhoods above threshold: "
      f"{hits}/{total} ({hits/total:.0%})"
    )
    top = nairobi.nlargest(5, "flood_prob")[["ward", "flood_prob"]]
    print("\nTop 5 Nairobi wards by model score (static features):")
    for _, r in top.iterrows():
      print(f"  {r['ward']}: {r['flood_prob']:.1%} ({risk_tier(r['flood_prob'], threshold)})")
  print()
  print("Sources cited per row in the table above.")


if __name__ == "__main__":
  main()
