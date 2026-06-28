#!/bin/bash
set -e

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp -n .env.example .env 2>/dev/null || true

echo ""
echo "Setup complete. Fill in your API keys in .env, then run:"
echo "  source .venv/bin/activate"
echo "  python main.py"
