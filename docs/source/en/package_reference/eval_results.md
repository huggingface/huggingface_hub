<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Evaluation Results

The Hub provides a decentralized system for tracking model evaluation results. Evaluation scores are stored in model repos as YAML files in the `.eval_results/` folder. These results automatically appear on the model page and are aggregated into benchmark dataset leaderboards.

For more details on the evaluation results format and benchmark integration, see the [Evaluation Results documentation](https://huggingface.co/docs/hub/eval-results).

## EvalResultEntry

[[autodoc]] EvalResultEntry

## eval_result_entries_to_yaml

[[autodoc]] huggingface_hub.eval_results.eval_result_entries_to_yaml

## parse_eval_result_entries

[[autodoc]] huggingface_hub.eval_results.parse_eval_result_entries
