<script>
	export let isDisabled = false;
	export let label = "Add Sentence";
	export let onClick: (e: MouseEvent) => void;
</script>

<button
	class="btn-widget w-full h-10 px-5"
	disabled={isDisabled}
	on:click|preventDefault={onClick}
	type="submit"
>
	{label}
</button>
