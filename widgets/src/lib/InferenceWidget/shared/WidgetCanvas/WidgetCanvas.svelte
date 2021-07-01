<script>
	import { width, heigth, canvas, img } from "./stores";
	export let imgSrc = "";
	export let mousemove: () => void = () => {};
	export let mouseout: () => void = () => {};
</script>

<div
	class="relative top-0 left-0"
	bind:clientWidth={$width}
	bind:clientHeight={$heigth}
>
	<img alt="" class="relative top-0 left-0" src={imgSrc} bind:this={$img} />
	<canvas
		class="absolute top-0 left-0"
		width={$width}
		height={$heigth}
		bind:this={$canvas}
		on:mousemove={mousemove}
		on:mouseout={mouseout}
	/>
</div>
