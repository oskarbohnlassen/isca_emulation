"""The public CLI must require user-supplied reproduction metadata."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/reproduce.py"


class ManifestTests(unittest.TestCase):
    def test_missing_manifest_stops_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = subprocess.run([sys.executable, str(SCRIPT), "all", "--root", tmp,
                "--manifest", str(root / "missing.json")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("Manifest not found", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse((root / "results").exists())

    def test_empty_template_is_rejected(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "all", "--manifest",
            str(ROOT / "scripts/manifest.example.json")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Manifest is incomplete", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_inventory_uses_the_supplied_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            outputs = {"figures": [{"output_png": "example.png"}]}
            path.write_text(json.dumps({"outputs": outputs}))
            result = subprocess.run([sys.executable, str(SCRIPT), "inventory", "--manifest", str(path)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), outputs)


if __name__ == "__main__":
    unittest.main()
