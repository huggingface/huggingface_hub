.PHONY: quality style test


check_dirs := tests src utils setup.py


quality:
	black --check $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)
	python utils/check_static_imports.py

style:
	black $(check_dirs)
	isort $(check_dirs)
	python utils/check_static_imports.py --update-file

test:
	HUGGINGFACE_CO_STAGING=1 pytest -sv ./tests/

