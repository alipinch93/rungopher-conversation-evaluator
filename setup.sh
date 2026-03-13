#!/bin/bash
# RunGopher Conversation Evaluator — Setup Script
# Run this from your terminal: bash setup.sh

cd "$(dirname "$0")"

echo "============================================"
echo "  RunGopher Conversation Evaluator Setup"
echo "============================================"

# 1. Create virtual environment
echo ""
echo "→ Creating virtual environment..."
rm -rf .venv
python3 -m venv .venv
if [ $? -ne 0 ]; then
    echo "⚠️  Could not create .venv here — trying ~/rg-eval-venv instead..."
    python3 -m venv ~/rg-eval-venv
    source ~/rg-eval-venv/bin/activate
    echo "→ Using ~/rg-eval-venv"
else
    source .venv/bin/activate
fi

# 2. Upgrade pip
echo "→ Upgrading pip..."
pip install --upgrade pip

# 3. Install Python dependencies
echo "→ Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt

# 4. Download spaCy NLP model for PII detection
echo "→ Downloading spaCy NLP model..."
python -m spacy download en_core_web_lg

# 5. Create output directories
echo "→ Creating output directories..."
mkdir -p data output/stripped output/summaries output/clusters output/reports

# 6. Check .env
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "⚠️  Created .env from template. Edit it to add your OpenAI API key:"
    echo "    nano .env"
else
    echo "→ .env file exists ✓"
fi

echo ""
echo "============================================"
echo "  Setup Complete! ✅"
echo "============================================"
echo ""
echo "  To start the evaluation server:"
echo "    source .venv/bin/activate"
echo "    python app.py"
echo ""
echo "  Then open: http://localhost:8000"
echo ""
echo "  Share this URL with your team (on the same network):"
echo "    http://$(hostname):8000"
echo ""
