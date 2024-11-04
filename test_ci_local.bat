@echo off
echo Running local CI checks...

echo.
echo === Installing dependencies ===
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
python -m spacy download en_core_web_sm
pip install -e .

echo.
echo === Running tests with pytest ===
pytest tests/ --cov=vision_framework --cov-report=xml || goto :error

echo.
echo === Running linting checks ===
black --check vision_framework tests || goto :error
isort --check-only vision_framework tests || goto :error
flake8 vision_framework tests || goto :error

echo.
echo === Building package ===
python -m build || goto :error

echo.
echo All CI checks passed successfully!
exit /b 0

:error
echo Failed with error #%errorlevel%.
exit /b %errorlevel%
