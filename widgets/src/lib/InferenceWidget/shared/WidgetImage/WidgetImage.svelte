<script>
	export let alt = "";
	export let classNames = "";
	export let imgSrc = "";
</script>

{#if imgSrc}
	<img {alt} class={classNames} src={imgSrc} />
{/if}
