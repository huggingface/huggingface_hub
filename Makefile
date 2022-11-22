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

# Taken from https://stackoverflow.com/a/12110773
# Commands:
#	make contrib_setup_timm : setup tests for timm
#	make contrib_test_timm  : run tests for timm
#	make contrib_timm       : setup and run tests for timm
#	make contrib_clear_timm : delete timm virtual env
#
#	make contrib_setup      : setup ALL tests
#	make contrib_test       : run ALL tests
#	make contrib            : setup and run ALL tests
#	make contrib_clear      : delete all virtual envs
# Use -j4 flag to run jobs in parallel.
CONTRIB_LIBS := sentence_transformers timm
CONTRIB_JOBS := $(addprefix contrib_,${CONTRIB_LIBS})
CONTRIB_CLEAR_JOBS := $(addprefix contrib_clear_,${CONTRIB_LIBS})
CONTRIB_SETUP_JOBS := $(addprefix contrib_setup_,${CONTRIB_LIBS})
CONTRIB_TEST_JOBS := $(addprefix contrib_test_,${CONTRIB_LIBS})

contrib_clear_%:
	rm -rf contrib/$*/.venv

contrib_setup_%:
	python3 -m venv contrib/$*/.venv
	./contrib/$*/.venv/bin/pip install -r contrib/requirements.txt
	./contrib/$*/.venv/bin/pip install -r contrib/$*/requirements.txt
	./contrib/$*/.venv/bin/pip uninstall -y huggingface_hub
	./contrib/$*/.venv/bin/pip install -e .

contrib_test_%:
	./contrib/$*/.venv/bin/python -m pytest contrib/$*

contrib_%:
	make contrib_setup_$*
	make contrib_test_$*

contrib: ${CONTRIB_JOBS};
contrib_clear: ${CONTRIB_CLEAR_JOBS}; echo "Successful contrib tests."
contrib_setup: ${CONTRIB_SETUP_JOBS}; echo "Successful contrib setup."
contrib_test: ${CONTRIB_TEST_JOBS}; echo "Successful contrib tests."