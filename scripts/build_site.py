"""Build the public website into docs/ (GitHub Pages serves main:/docs).

Runs the model-output export, inlines it into the template, and writes
docs/index.html. Run after any retrain, then commit + push to update the
live site.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

r = subprocess.run([sys.executable, str(ROOT / "scripts" / "export_web.py")], cwd=ROOT)
if r.returncode != 0:
    sys.exit("export failed")

tpl = (ROOT / "web" / "index.template.html").read_text()
data = (ROOT / "web" / "data.json").read_text()
assert "/*__DATA__*/" in tpl, "template placeholder missing"
out = ROOT / "docs" / "index.html"
out.parent.mkdir(exist_ok=True)
out.write_text(tpl.replace("/*__DATA__*/", data))
(ROOT / "web" / "index.html").write_text(tpl.replace("/*__DATA__*/", data))
print(f"built {out} ({out.stat().st_size/1024:.0f} KB)")
print("publish: git add docs && git commit -m 'update site' && git push")
