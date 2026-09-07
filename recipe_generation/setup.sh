#!/bin/bash
set -e

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp -n .env.example .env 2>/dev/null || true

echo ""
echo "Setup complete. Put your OpenAI key in .env, then run:"
echo "  source .venv/bin/activate"
echo "  python test_scoring.py && python generate.py && python extract.py && python validate_extraction.py && python score.py && python analyze.py"
