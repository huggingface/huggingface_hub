.PHONY: contrib quality style test


check_dirs := contrib src tests utils setup.py


quality:
	black --check $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)
	mypy src
	python utils/check_static_imports.py

style:
	black $(check_dirs)
	isort $(check_dirs)
	python utils/check_static_imports.py --update-file

test:
	pytest ./tests/

contrib:
	python3 -m venv contrib/timm/.venv
	. contrib/timm/.venv/bin/activate
	pip install -r contrib/requirements.txt
	pip install -r contrib/timm/requirements.txt
	pip uninstall -y huggingface_hub
	pip install -e .
	pytest contrib/timm