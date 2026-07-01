"""Contract tests for the bundled telemetry skills."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
SKILLS = (
    ROOT / "skills" / "devops" / "telemetry" / "SKILL.md",
    ROOT / "skills" / "devops" / "telemetry-analysis" / "SKILL.md",
)
REQUIRED_SECTIONS = (
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
)


def test_telemetry_skills_follow_bundled_skill_contract():
    for path in SKILLS:
        source = path.read_text(encoding="utf-8")
        description = re.search(r'^description: "(.*)"$', source, re.MULTILINE)
        assert description and len(description.group(1)) <= 60
        assert 'author: "Will Killian (@willkill07), Hermes Agent"' in source
        assert "platforms: [linux, macos, windows]" in source
        positions = [source.index(section) for section in REQUIRED_SECTIONS]
        assert positions == sorted(positions)
