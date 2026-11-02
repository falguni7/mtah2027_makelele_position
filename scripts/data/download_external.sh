#!/usr/bin/env bash
# Download the public external datasets (other matches, not BAR-ATH) into external/.
#
#   bash scripts/data/download_external.sh            # all three, ~6.5 GB
#
#   external/dfl/          DFL Bundesliga open data, 7 matches, CC BY 4.0 (figshare 28196177)
#                          Bassek et al., Scientific Data 12, 195 (2025)
#   external/metrica/      Metrica Sports sample data, 3 matches
#   external/skillcorner/  SkillCorner open data, 10 A-League matches, MIT
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/external"
mkdir -p "$OUT/dfl"

[ -d "$OUT/metrica" ] || git clone --depth 1 https://github.com/metrica-sports/sample-data.git "$OUT/metrica"
[ -d "$OUT/skillcorner" ] || git clone --depth 1 https://github.com/SkillCorner/opendata.git "$OUT/skillcorner"

# figshare API: one line per file, "<download_url> <name> <size>"; skip files already complete.
curl -s https://api.figshare.com/v2/articles/28196177 | python3 -c '
import json, sys
for f in json.load(sys.stdin)["files"]:
    print(f["download_url"], f["name"], f["size"])' |
while read -r url name size; do
    if [ -f "$OUT/dfl/$name" ] && [ "$(stat -c%s "$OUT/dfl/$name")" = "$size" ]; then
        continue
    fi
    echo "downloading $name"
    curl -sL -o "$OUT/dfl/$name" "$url"
done
echo "done: $(du -sh "$OUT" | cut -f1) in $OUT"
