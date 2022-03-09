pip install -U pip build twine
python -m build
python -m twine upload dist/*

