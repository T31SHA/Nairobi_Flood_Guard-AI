"""CAP-aligned Alert object tests."""

import json
from xml.etree import ElementTree as ET

from Utils.alerts import (
    Certainty,
    Severity,
    Urgency,
    build_alert,
    prob_to_severity,
    risk_tier,
)

REGISTRY_THRESHOLD = 0.2967171718676885


class TestSeverityMapping:
    def test_at_registry_threshold_is_moderate(self):
        assert prob_to_severity(REGISTRY_THRESHOLD, REGISTRY_THRESHOLD) == Severity.MODERATE

    def test_critical_maps_extreme(self):
        assert prob_to_severity(0.75, REGISTRY_THRESHOLD) == Severity.EXTREME

    def test_high_maps_severe(self):
        assert prob_to_severity(0.50, REGISTRY_THRESHOLD) == Severity.SEVERE

    def test_low_maps_minor(self):
        assert prob_to_severity(0.10, REGISTRY_THRESHOLD) == Severity.MINOR

    def test_risk_tier_boundaries(self):
        assert risk_tier(0.75, REGISTRY_THRESHOLD) == "Critical"
        assert risk_tier(0.50, REGISTRY_THRESHOLD) == "High"
        assert risk_tier(0.30, REGISTRY_THRESHOLD) == "Moderate"
        assert risk_tier(0.05, REGISTRY_THRESHOLD) == "Low"


class TestAlertSerialization:
    def test_cap_xml_structure(self):
        alert = build_alert(
            "Ruai Ward", "Nairobi", 0.67, REGISTRY_THRESHOLD, kind="new"
        )
        xml = alert.to_cap_xml()
        root = ET.fromstring(xml)
        ns = {"cap": "urn:oasis:names:tc:emergency:cap:1.2"}
        assert root.tag.endswith("alert")
        info = root.find("cap:info", ns)
        assert info is not None
        assert info.find("cap:severity", ns).text == "Severe"
        assert info.find("cap:urgency", ns).text == "Immediate"
        assert info.find("cap:certainty", ns).text == "Observed"
        assert info.find("cap:description", ns).text
        assert info.find("cap:instruction", ns).text
        area = info.find("cap:area", ns)
        assert area.find("cap:areaDesc", ns).text

    def test_json_roundtrip_fields(self):
        alert = build_alert("Kayole Central Ward", "Nairobi", 0.4, REGISTRY_THRESHOLD)
        data = json.loads(alert.to_json())
        assert data["severity"] == "Moderate"
        assert "sms" in data
        assert data["description"]

    def test_sms_english_and_swahili(self):
        alert = build_alert(
            "Mathare North Ward", "Nairobi", 0.35, REGISTRY_THRESHOLD, language="en"
        )
        assert "FLOOD ALERT" in alert.to_sms("en")
        assert "MAFURIKO" in alert.to_sms("sw")

    def test_horizon_affects_urgency(self):
        live = build_alert("Ruai Ward", "Nairobi", 0.5, REGISTRY_THRESHOLD, horizon_hours=0)
        forecast = build_alert("Ruai Ward", "Nairobi", 0.5, REGISTRY_THRESHOLD, horizon_hours=48)
        assert live.urgency == Urgency.IMMEDIATE
        assert forecast.urgency == Urgency.FUTURE
        assert forecast.certainty == Certainty.POSSIBLE
