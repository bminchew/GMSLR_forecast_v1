#!/bin/sh
cd "$(dirname "$0")"
python3 gen.py
C="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
"$C" --headless --disable-gpu --hide-scrollbars --virtual-time-budget=4000 --screenshot=$PWD/figure.png --window-size=1344,880 file://$PWD/figure.html 2>/dev/null
