#!/usr/bin/env bash
# Download the public external datasets (other matches, not BAR-ATH) into external/.
#
# These are labelled matches from other games, for later steps (BAR-ATH data comes first). They
# are public, but external/ is gitignored all the same: they are large and fetched by this script.
#
#   bash scripts/data/download_external.sh            # all three, ~6.2 G as reported by du -sh
#
#   external/dfl/          DFL Bundesliga open data, 7 matches, CC BY 4.0 (figshare 28196177)
#                          Bassek et al., Scientific Data 12, 195 (2025)
#   external/metrica/      Metrica Sports sample data, 3 matches
#   external/skillcorner/  SkillCorner open data, 10 A-League matches, MIT
#
# Safe to run again, with one caveat: a git clone is skipped as soon as its folder exists, so an
# interrupted clone is not repaired (delete the folder to retry). Only the DFL files are checked
# by size: one already on disk with the size figshare lists is not downloaded again (an
# interrupted one is). Run it on a node with
# internet access (needs git, curl and python3); it is a download, not a compute job, so it does
# not go through run.sbatch.
#
# Output: on a first run git clone's own "Cloning into ..." lines, then one "downloading <file>"
# line per DFL file fetched, then "done: <total size> in <external folder>".
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/external"
mkdir -p "$OUT/dfl"

[ -d "$OUT/metrica" ] || git clone --depth 1 https://github.com/metrica-sports/sample-data.git "$OUT/metrica"
[ -d "$OUT/skillcorner" ] || git clone --depth 1 https://github.com/SkillCorner/opendata.git "$OUT/skillcorner"

# figshare API: one line per file, "<download_url> <name> <size>"; skip files already complete.
# Complete means: present and exactly the listed size in bytes.
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
