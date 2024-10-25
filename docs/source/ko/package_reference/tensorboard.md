<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# TensorBoard 로거[[tensorboard-logger]]

TensorBoard는 기계학습 실험을 위한 시각화 도구입니다. 주로 손실 및 정확도와 같은 지표를 추적 및 시각화하고, 모델 그래프와 
히스토그램을 보여주고, 이미지를 표시하는 등 다양한 기능을 제공합니다. 또한 TensorBoard는 Hugging Face Hub와 잘 통합되어 있습니다. 
`tfevents` 같은 TensorBoard 추적을 Hub에 푸시하면 Hub는 이를 자동으로 감지하여 시각화 인스턴스를 시작합니다.
TensorBoard와 Hub의 통합에 대한 자세한 정보는 [가이드](https://huggingface.co/docs/hub/tensorboard)를 확인하세요.

이 통합을 위해, `huggingface_hub`는 로그를 Hub로 푸시하기 위한 사용자 정의 로거를 제공합니다. 
이 로거는 추가적인 코드 없이 [SummaryWriter](https://tensorboardx.readthedocs.io/en/latest/tensorboard.html)의 대체제로 사용될 수 있습니다. 
추적은 계속해서 로컬에 저장되며 백그라운드 작업이 일정한 시간마다 Hub에 푸시하는 형태로 동작합니다.

## HFSummaryWriter[[huggingface_hub.HFSummaryWriter]]

[[autodoc]] HFSummaryWriter
