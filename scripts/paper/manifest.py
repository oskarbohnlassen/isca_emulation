"""Load user-supplied metadata for figure reproduction."""
import json
from pathlib import Path


class ManifestError(ValueError):
    pass


_manifest = None

COMMON = ("selected_checkpoints", "input_hashes")
REQUIRED = {
    "inventory": ("outputs",),
    "tables": (*COMMON, "controlled_checkpoints", "selected_hyperparameters", "events"),
    "profiles": COMMON,
    "spatial": COMMON,
    "persistence": COMMON,
    "temporal": COMMON,
    "ep-main": COMMON,
    "ep-si": (*COMMON, "events"),
    "verify": ("outputs", "input_hashes", "events", "source_provenance"),
    "all": (*COMMON, "controlled_checkpoints", "selected_hyperparameters", "events", "outputs", "source_provenance"),
}


def configure_manifest(path, command):
    global _manifest
    path = Path(path)
    guidance = ("Copy scripts/manifest.example.json to scripts/manifest.json, "
                "fill in the required metadata, and provide the matching evaluated data. "
                "Populated manifests and data are not distributed with this repository.")
    if not path.is_file():
        raise ManifestError(f"Manifest not found: {path}. {guidance}")
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ManifestError(f"Cannot read manifest: {path}. {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"Manifest must be a JSON object. {guidance}")
    missing = []
    contents = {"selected_checkpoints": "runs", "controlled_checkpoints": "runs",
                "selected_hyperparameters": "records", "events": "events",
                "outputs": "figures", "source_provenance": "functions"}
    for section in REQUIRED[command]:
        record = value.get(section)
        key = contents.get(section)
        if not isinstance(record, dict) or not (record.get(key) if key else record):
            missing.append(section)
        elif section.endswith("checkpoints") and record.get("verified") is not True:
            missing.append(section)
    if missing:
        raise ManifestError(f"Manifest is incomplete: {', '.join(missing)}. {guidance}")
    _manifest = value


def read_manifest(name):
    if _manifest is None:
        raise ManifestError("Configure a populated manifest before running the analysis.")
    section = Path(name).stem
    try:
        return _manifest[section]
    except KeyError as exc:
        raise ManifestError(f"Required manifest section is missing: {section}") from exc
