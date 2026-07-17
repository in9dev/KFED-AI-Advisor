#!/usr/bin/env bash
# No dependencies to install — pure Python standard library.
set -e
cd "$(dirname "$0")"
python3 -m backend.app
ngrok http 8420
