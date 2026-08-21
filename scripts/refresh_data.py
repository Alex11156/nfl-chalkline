"""One-command refresh: re-download latest data, rebuild the store and
features, retrain final models. Run this during the season to keep the
engine current (takes ~15-30 min).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

STEPS = [
    [PY, str(ROOT / "scripts" / "download_data.py"), "--refresh"],
    [PY, "-m", "nfl_engine.data.store"],
    [PY, "-m", "nfl_engine.features.game_features"],
    [PY, "-m", "nfl_engine.features.player_features"],
    [PY, "-m", "nfl_engine.models.game_models"],
    [PY, "-m", "nfl_engine.models.player_models"],
    [PY, str(ROOT / "scripts" / "recalibrate.py")],
    [PY, str(ROOT / "scripts" / "build_site.py")],
]

for step in STEPS:
    print(">>", " ".join(step), flush=True)
    r = subprocess.run(step, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"step failed: {' '.join(step)}")
print("refresh complete — site rebuilt in docs/")
if "--publish" in sys.argv:
    for cmd in [["git", "add", "docs"],
                ["git", "commit", "-m", "site: data refresh"],
                ["git", "push"]]:
        subprocess.run(cmd, cwd=ROOT)
else:
    print("to publish the site: rerun with --publish, or "
          "git add docs && git commit -m 'update site' && git push")
