#!/bin/bash
# One-time GitHub Pages deployment. Prerequisite (interactive, ~1 min):
#   /opt/anaconda3/envs/nfl/bin/gh auth login
# Then run:
#   bash scripts/deploy_github.sh
set -e
GH=/opt/anaconda3/envs/nfl/bin/gh
cd "$(dirname "$0")/.."

$GH auth status >/dev/null 2>&1 || { echo "Run: $GH auth login   (choose GitHub.com, HTTPS, login via browser)"; exit 1; }

REPO=nfl-chalkline
$GH repo create "$REPO" --public --source . --push --description "NFL prediction engine + Chalkline site"
OWNER=$($GH api user --jq .login)
# Enable Pages serving main:/docs
$GH api "repos/$OWNER/$REPO/pages" -X POST -f 'source[branch]=main' -f 'source[path]=/docs' 2>/dev/null \
  || $GH api "repos/$OWNER/$REPO/pages" -X PUT -f 'source[branch]=main' -f 'source[path]=/docs'
echo ""
echo "✅ Site will be live in ~1 minute at:"
echo "   https://$OWNER.github.io/$REPO/"
