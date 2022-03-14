.PHONY: quality style test


check_dirs := tests src


quality:
	black --check $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)

style:
	black $(check_dirs)
	isort $(check_dirs)

test:
	HUGGINGFACE_CO_STAGING=1 pytest -sv ./tests/

