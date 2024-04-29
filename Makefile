.PHONY: contrib quality style test


check_dirs := contrib src tests utils setup.py


quality:
	ruff check $(check_dirs)  # linter
	ruff format --check $(check_dirs) # formatter
	python utils/check_inference_input_params.py
	python utils/check_contrib_list.py
	python utils/check_static_imports.py
	python utils/generate_async_inference_client.py
	mypy src

style:
	ruff format $(check_dirs) # formatter
	ruff check --fix $(check_dirs) # linter
	python utils/check_contrib_list.py --update
	python utils/check_static_imports.py --update
	python utils/generate_async_inference_client.py --update

inference_types_check:
	python utils/generate_inference_types.py

inference_types_update:
	python utils/generate_inference_types.py --update

repocard:
	python utils/push_repocard_examples.py


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
CONTRIB_LIBS := sentence_transformers spacy timm
CONTRIB_JOBS := $(addprefix contrib_,${CONTRIB_LIBS})
CONTRIB_CLEAR_JOBS := $(addprefix contrib_clear_,${CONTRIB_LIBS})
CONTRIB_SETUP_JOBS := $(addprefix contrib_setup_,${CONTRIB_LIBS})
CONTRIB_TEST_JOBS := $(addprefix contrib_test_,${CONTRIB_LIBS})

contrib_clear_%:
	rm -rf contrib/$*/.venv

contrib_setup_%:
	python3 -m venv contrib/$*/.venv
	./contrib/$*/.venv/bin/pip install -r contrib/$*/requirements.txt
	./contrib/$*/.venv/bin/pip uninstall -y huggingface_hub
	./contrib/$*/.venv/bin/pip install -e .[testing]

contrib_test_%:
	./contrib/$*/.venv/bin/python -m pytest contrib/$*

contrib_%:
	make contrib_setup_$*
	make contrib_test_$*

contrib: ${CONTRIB_JOBS};
contrib_clear: ${CONTRIB_CLEAR_JOBS}; echo "Successful contrib tests."
contrib_setup: ${CONTRIB_SETUP_JOBS}; echo "Successful contrib setup."
contrib_test: ${CONTRIB_TEST_JOBS}; echo "Successful contrib tests."
