#!/bin/bash

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Download required NLTK data
python -c "import nltk; nltk.download('wordnet')"

# Download spaCy model
python -m spacy download en_core_web_sm

echo "Development environment setup complete!"
