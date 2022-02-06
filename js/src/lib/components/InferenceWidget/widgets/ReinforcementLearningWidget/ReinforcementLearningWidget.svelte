<script>
	import type { WidgetProps } from "../../shared/types";

	import WidgetWrapper from "../../shared/WidgetWrapper/WidgetWrapper.svelte";

	export let apiUrl: WidgetProps["apiUrl"];
	export let model: WidgetProps["model"];

	let error: string = "";
	let computeTime = "";
	let isLoading = false;
	let modelLoading = {
		isLoading: false,
		estimatedTime: 0,
	};
	let outputJson: string;
</script>

<WidgetWrapper
	{apiUrl}
	{computeTime}
	{error}
	{isLoading}
	{model}
	{modelLoading}
	noTitle={true}
	noModelInfo={true}
	{outputJson}
>
	<svelte:fragment slot="top">
		<!-- svelte-ignore a11y-media-has-caption -->
		<video
			src="https://huggingface.co/ThomasSimonini/ppo-LunarLander-v2/resolve/main/output.mp4"
			controls
			autoplay
			loop
		/>
		<div class="flex justify-center">
			<img
				alt="clipart game controller"
				class="dark:invert w-36"
				src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Video_game_controller_icon_designed_by_Maico_Amorim.svg/1249px-Video_game_controller_icon_designed_by_Maico_Amorim.svg.png"
			/>
		</div>
	</svelte:fragment>
</WidgetWrapper>
