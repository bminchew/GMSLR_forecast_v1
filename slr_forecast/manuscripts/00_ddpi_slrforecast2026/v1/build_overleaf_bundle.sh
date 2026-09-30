#!/usr/bin/env bash
# Build a self-contained Overleaf upload bundle for the main text and supplement.
#
# Copies both .tex files, references.bib, and agufull08.bst into overleaf/,
# copies every figure the two documents include (commented-out calls are
# skipped) into overleaf/figures/, points \graphicspath at figures/ in the
# copies, test-compiles both, and zips the result as overleaf.zip.
# The .tex files in this directory are not modified.
#
# Usage: ./build_overleaf_bundle.sh     (from any directory)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
FIG_SRC="$HERE/../../../figures"
OUT="$HERE/overleaf"
DOCS=(01_slr_forecast_intervention2026 01_slr_forecast_intervention2026_supp)

rm -rf "$OUT" "$HERE/overleaf.zip"
mkdir -p "$OUT/figures"
cp "$HERE/references.bib" "$HERE/agufull08.bst" "$OUT/"

for doc in "${DOCS[@]}"; do
    # Point the copy at the bundled figures folder.
    sed 's|\\graphicspath{{../../../figures/}}|\\graphicspath{{figures/}}|' \
        "$HERE/$doc.tex" > "$OUT/$doc.tex"
    grep -q '\\graphicspath{{figures/}}' "$OUT/$doc.tex" \
        || { echo "graphicspath not rewritten in $doc.tex" >&2; exit 1; }

    # Figures included on live (uncommented) lines.
    sed 's/\([^\\]\)%.*$/\1/; s/^%.*$//' "$HERE/$doc.tex" \
        | grep -o 'includegraphics\(\[[^]]*\]\)\{0,1\}{[^}]*}' \
        | sed 's/.*{//; s/}$//' \
        | while read -r fig; do
            [ -f "$FIG_SRC/$fig" ] || { echo "missing figure: $fig" >&2; exit 1; }
            cp "$FIG_SRC/$fig" "$OUT/figures/"
        done
done

# Test-compile both documents from the bundle, then remove build products.
TMP="$(mktemp -d)"
cp -R "$OUT/." "$TMP/"
for doc in "${DOCS[@]}"; do
    (cd "$TMP" && latexmk -pdf -interaction=nonstopmode -halt-on-error "$doc.tex" > /dev/null 2>&1) \
        || { echo "compile failed: $doc (see $TMP/$doc.log)" >&2; exit 1; }
    if grep -qiE 'undefined|File .* not found' "$TMP/$doc.log"; then
        echo "warnings in $doc.log:" >&2
        grep -iE 'undefined|File .* not found' "$TMP/$doc.log" | sort -u >&2
        exit 1
    fi
done
rm -rf "$TMP"

(cd "$OUT" && zip -qr "$HERE/overleaf.zip" .)
echo "Bundle: $OUT ($(ls "$OUT/figures" | wc -l | tr -d ' ') figures)"
echo "Zip:    $HERE/overleaf.zip ($(du -h "$HERE/overleaf.zip" | cut -f1))"
