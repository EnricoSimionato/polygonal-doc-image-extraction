#!/bin/bash
# Starts the app (macOS and Linux). First run: creates .venv and installs the requirements (a few minutes, needs
# internet) and downloads the model weights (93 MB) from the v0 release. Then it opens the browser.
cd "$(dirname "$0")" || exit 1
# a Python 3.10 or newer: the PATH first, then the usual install locations (a double-clicked file on macOS does not
# load the shell profile, so conda and Homebrew may be missing from the PATH)
PY=""
for p in python3.12 python3.11 python3.13 python3.10 python3 \
         "$HOME/miniconda3/bin/python3" "$HOME/anaconda3/bin/python3" "$HOME/miniforge3/bin/python3" \
         /opt/homebrew/bin/python3 /usr/local/bin/python3 \
         /Library/Frameworks/Python.framework/Versions/3.1[0-9]/bin/python3; do
  c="$(command -v "$p" 2>/dev/null || true)"; [ -z "$c" ] && [ -x "$p" ] && c="$p"
  if [ -n "$c" ] && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done
[ -z "$PY" ] && { echo "Python 3.10 or newer is required: https://www.python.org/downloads/"; read -r -p "Press Enter to close."; exit 1; }
REQ=$(shasum requirements.txt 2>/dev/null | cut -c1-12 || sha1sum requirements.txt | cut -c1-12)
if [ ! -f .venv/.ready ] || [ "$(cat .venv/.ready)" != "$REQ" ]; then
  echo "Setting up the Python environment in .venv (first run only)…"
  [ -x .venv/bin/python ] || "$PY" -m venv .venv || exit 1
  .venv/bin/python -m pip install -q --upgrade pip && .venv/bin/python -m pip install -q -r requirements.txt \
    || { read -r -p "Installation failed (internet connection needed). Press Enter to close."; exit 1; }
  echo "$REQ" > .venv/.ready
fi
exec .venv/bin/python app/server.py
