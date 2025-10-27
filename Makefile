.PHONY: quality style test


check_dirs := src tests utils setup.py


quality:
	ruff check $(check_dirs)  # linter
	ruff format --check $(check_dirs) # formatter
	python utils/check_inference_input_params.py
	python utils/check_static_imports.py
	python utils/check_all_variable.py
	python utils/generate_async_inference_client.py
	python utils/generate_cli_reference.py

	ty check src

style:
	ruff format $(check_dirs) # formatter
	ruff check --fix $(check_dirs) # linter
	python utils/check_static_imports.py --update
	python utils/check_all_variable.py --update
	python utils/generate_async_inference_client.py --update
	python utils/generate_cli_reference.py --update

inference_check:
	python utils/generate_inference_types.py
	python utils/check_task_parameters.py

inference_update:
	python utils/generate_inference_types.py --update
	python utils/check_task_parameters.py --update


repocard:
	python utils/push_repocard_examples.py


test:
	pytest ./tests/
