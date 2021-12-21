<script lang="ts">
	import type { IconData } from "./Types";

	export let classNames = "";
	export let data: IconData;
</script>

<svg
	aria-hidden="true"
	class={classNames}
	fill={data.fill ?? "currentColor"}
	focusable="false"
	height={data.height ?? "1em"}
	preserveAspectRatio="xMidYMid meet"
	role="img"
	style={data.style}
	viewBox={data.viewbox ?? "0 0 32 32"}
	width={data.width ?? "1em"}
	xmlns="http://www.w3.org/2000/svg"
	xmlns:xlink="http://www.w3.org/1999/xlink"
>
	{@html data.body}
</svg>
