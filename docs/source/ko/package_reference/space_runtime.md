<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Space 런타임 관리[[managing-your-space-runtime]]

Hub의 Space를 관리하는 메소드에 대한 자세한 설명은 [`HfApi`]페이지를 확인하세요.

- Space 복제: [`duplicate_space`]
- 현재 런타임 가져오기: [`get_space_runtime`]
- 보안 관리: [`add_space_secret`] 및 [`delete_space_secret`]
- 하드웨어 관리: [`request_space_hardware`]
- 상태 관리: [`pause_space`], [`restart_space`], [`set_space_sleep_time`]

## 데이터 구조[[data-structures]]

### SpaceRuntime[[huggingface_hub.SpaceRuntime]]

[[autodoc]] SpaceRuntime

### SpaceHardware[[huggingface_hub.SpaceHardware]]

[[autodoc]] SpaceHardware

### SpaceStage[[huggingface_hub.SpaceStage]]

[[autodoc]] SpaceStage

### SpaceStorage[[huggingface_hub.SpaceStorage]]

[[autodoc]] SpaceStorage

### SpaceVariable[[huggingface_hub.SpaceVariable]]

[[autodoc]] SpaceVariable