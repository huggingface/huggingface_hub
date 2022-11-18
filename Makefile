.PHONY: quality style test


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
