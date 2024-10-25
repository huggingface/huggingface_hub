<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# TensorBoard logger

TensorBoard is a visualization toolkit for machine learning experimentation. TensorBoard allows tracking and visualizing
metrics such as loss and accuracy, visualizing the model graph, viewing histograms, displaying images and much more.
TensorBoard is well integrated with the Hugging Face Hub. The Hub automatically detects TensorBoard traces (such as
`tfevents`) when pushed to the Hub which starts an instance to visualize them. To get more information about TensorBoard
integration on the Hub, check out [this guide](https://huggingface.co/docs/hub/tensorboard).

To benefit from this integration, `huggingface_hub` provides a custom logger to push logs to the Hub. It works as a
drop-in replacement for [SummaryWriter](https://tensorboardx.readthedocs.io/en/latest/tensorboard.html) with no extra
code needed. Traces are still saved locally and a background job push them to the Hub at regular interval.

## HFSummaryWriter

[[autodoc]] HFSummaryWriter
