"""CAP-aligned flood alerts for Nairobi Flood Guard.

Every warning is a structured ``Alert`` object with the same conceptual fields
as OASIS CAP 1.2 (severity, urgency, certainty, area, effective/onset/expiry,
description, instruction). Channel formatters (SMS, WhatsApp, JSON feed, CAP XML)
all render from this single object — no channel-specific alert logic.

Severity mapping (from calibrated ``flood_prob`` and registry threshold):
    Critical (>= 0.70)  -> Extreme
    High     (>= 0.45)  -> Severe
    Moderate (>= threshold) -> Moderate
    Low      (< threshold)  -> Minor

The registry threshold is the Moderate/High boundary for early warning.
Urgency derives from forecast horizon; certainty from whether the crossing
is live/observed vs forecast.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Literal
from xml.sax.saxutils import escape

Language = Literal["en", "sw"]

SENDER = "nairobi-flood-guard@kenya.local"
SENDER_NAME = "Nairobi Flood Guard (complements KMD & Kenya Red Cross)"


class Severity(StrEnum):
    EXTREME = "Extreme"
    SEVERE = "Severe"
    MODERATE = "Moderate"
    MINOR = "Minor"


class Urgency(StrEnum):
    IMMEDIATE = "Immediate"
    EXPECTED = "Expected"
    FUTURE = "Future"
    PAST = "Past"


class Certainty(StrEnum):
    OBSERVED = "Observed"
    LIKELY = "Likely"
    POSSIBLE = "Possible"
    UNLIKELY = "Unlikely"


@dataclass
class Alert:
  event: str = "Flood"
  severity: Severity = Severity.MODERATE
  urgency: Urgency = Urgency.IMMEDIATE
  certainty: Certainty = Certainty.LIKELY
  area: str = ""
  county: str = ""
  ward: str = ""
  flood_prob: float = 0.0
  effective: datetime = field(default_factory=lambda: datetime.now(UTC))
  onset: datetime = field(default_factory=lambda: datetime.now(UTC))
  expires: datetime = field(default_factory=lambda: datetime.now(UTC) + timedelta(hours=24))
  description_en: str = ""
  description_sw: str = ""
  instruction_en: str = ""
  instruction_sw: str = ""
  identifier: str = field(default_factory=lambda: str(uuid.uuid4()))
  language: Language = "en"
  horizon_hours: int = 0
  kind: str = "new"  # new | escalation

  def description(self, language: Language | None = None) -> str:
    lang = language or self.language
    return self.description_sw if lang == "sw" else self.description_en

  def instruction(self, language: Language | None = None) -> str:
    lang = language or self.language
    return self.instruction_sw if lang == "sw" else self.instruction_en

  def to_sms(self, language: Language | None = None) -> str:
    lang = language or self.language
    sev = self.severity.value.upper()
    prefix = "TAHARIRI YA MAFURIKO" if lang == "sw" else "FLOOD ALERT"
    if self.kind == "escalation":
      prefix += " (HATARI KUBWA)" if lang == "sw" else " (CRITICAL)"
    area = f"{self.ward} ({self.county})" if self.ward else self.area
    body = self.description(lang)
    instr = self.instruction(lang)
    footer = "- Nairobi Flood Guard"
    msg = f"{prefix}: {area} [{sev}] {body} {instr} {footer}"
    return msg[:640]  # WhatsApp-friendly cap; SMS may split across pages

  def to_dict(self) -> dict:
    d = asdict(self)
    d["severity"] = self.severity.value
    d["urgency"] = self.urgency.value
    d["certainty"] = self.certainty.value
    d["effective"] = self.effective.isoformat()
    d["onset"] = self.onset.isoformat()
    d["expires"] = self.expires.isoformat()
    d["description"] = self.description()
    d["instruction"] = self.instruction()
    d["sms"] = self.to_sms()
    return d

  def to_json(self) -> str:
    return json.dumps(self.to_dict(), ensure_ascii=False)

  def to_cap_xml(self) -> str:
    """Minimal CAP 1.2 XML for interoperability (not full network integration)."""
    desc = escape(self.description_en)
    instr = escape(self.instruction_en)
    area = escape(self.area or f"{self.ward}, {self.county}")
    headline = escape(
      f"{self.event} - {self.severity.value} - {self.ward or self.area}"
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>{escape(self.identifier)}</identifier>
  <sender>{escape(SENDER)}</sender>
  <sent>{self.effective.strftime("%Y-%m-%dT%H:%M:%S")}Z</sent>
  <status>Actual</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <info>
    <language>en-KE</language>
    <category>Met</category>
    <event>{escape(self.event)}</event>
    <urgency>{self.urgency.value}</urgency>
    <severity>{self.severity.value}</severity>
    <certainty>{self.certainty.value}</certainty>
    <effective>{self.effective.strftime("%Y-%m-%dT%H:%M:%S")}Z</effective>
    <onset>{self.onset.strftime("%Y-%m-%dT%H:%M:%S")}Z</onset>
    <expires>{self.expires.strftime("%Y-%m-%dT%H:%M:%S")}Z</expires>
    <senderName>{escape(SENDER_NAME)}</senderName>
    <headline>{headline}</headline>
    <description>{desc}</description>
    <instruction>{instr}</instruction>
    <area>
      <areaDesc>{area}</areaDesc>
    </area>
  </info>
</alert>"""


def risk_tier(prob: float, threshold: float) -> str:
  if prob >= 0.70:
    return "Critical"
  if prob >= 0.45:
    return "High"
  if prob >= threshold:
    return "Moderate"
  return "Low"


def prob_to_severity(prob: float, threshold: float) -> Severity:
  tier = risk_tier(prob, threshold)
  return {
    "Critical": Severity.EXTREME,
    "High": Severity.SEVERE,
    "Moderate": Severity.MODERATE,
    "Low": Severity.MINOR,
  }[tier]


def horizon_to_urgency(horizon_hours: int) -> Urgency:
  if horizon_hours <= 0:
    return Urgency.IMMEDIATE
  if horizon_hours <= 24:
    return Urgency.EXPECTED
  return Urgency.FUTURE


def horizon_to_certainty(horizon_hours: int, kind: str) -> Certainty:
  if horizon_hours <= 0:
    return Certainty.OBSERVED if kind == "new" else Certainty.LIKELY
  if horizon_hours <= 24:
    return Certainty.LIKELY
  return Certainty.POSSIBLE


def build_alert(
  ward: str,
  county: str,
  flood_prob: float,
  threshold: float,
  *,
  prev_prob: float | None = None,
  kind: str = "new",
  horizon_hours: int = 0,
  language: Language = "en",
) -> Alert:
  """Map a ward crossing into a CAP-shaped Alert."""
  sev = prob_to_severity(flood_prob, threshold)
  urgency = horizon_to_urgency(horizon_hours)
  certainty = horizon_to_certainty(horizon_hours, kind)
  now = datetime.now(UTC)
  tier = risk_tier(flood_prob, threshold)

  was_en = f" (was {prev_prob:.0%})" if prev_prob is not None else ""
  was_sw = f" (ilikuwa {prev_prob:.0%})" if prev_prob is not None else ""

  if kind == "escalation":
    desc_en = (
      f"Flood risk in {ward} ({county}) has escalated to {flood_prob:.0%} "
      "- in the critical band."
    )
    instr_en = (
      "Avoid low-lying areas and flooded routes. Follow local authority and "
      "Kenya Red Cross guidance."
    )
    desc_sw = (
      f"Hatari ya mafuriko katika {ward} imeongezeka hadi {flood_prob:.0%} "
      f"- kiwango cha hatari kubwa."
    )
    instr_sw = (
      "Epuka maeneo ya chini na barabara zilizofurika. Fuata maagizo ya "
      "mamlaka za mitaa na Kenya Red Cross."
    )
  else:
    desc_en = (
      f"Flood risk in {ward} ({county}) is now {flood_prob:.0%}, above the "
      f"{threshold:.0%} early-warning threshold{was_en}. Tier: {tier}."
    )
    instr_en = (
      "Avoid low-lying areas and flooded routes. Follow official KMD advisories."
    )
    desc_sw = (
      f"Hatari ya mafuriko katika {ward} ({county}) sasa ni {flood_prob:.0%}, "
      f"juu ya kizingiti cha {threshold:.0%}{was_sw}. Kiwango: {tier}."
    )
    instr_sw = (
      "Epuka maeneo ya chini na njia zilizofurika. Fuata taarifa rasmi za KMD."
    )

  return Alert(
    severity=sev,
    urgency=urgency,
    certainty=certainty,
    area=f"{ward}, {county} County, Kenya",
    county=county,
    ward=ward,
    flood_prob=flood_prob,
    effective=now,
    onset=now,
    expires=now + timedelta(hours=24 if horizon_hours <= 24 else 48),
    description_en=desc_en,
    description_sw=desc_sw,
    instruction_en=instr_en,
    instruction_sw=instr_sw,
    language=language,
    horizon_hours=horizon_hours,
    kind=kind,
  )
