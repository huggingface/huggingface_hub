<script>
	export let position: "left" | "right";
	export let text: string;
</script>

<div
	class="px-3 py-2 rounded-2xl {position === 'right'
		? 'bg-blue-600 text-white ml-7'
		: 'place-self-start bg-gray-50 dark:bg-gray-850 dark:text-gray-200 mr-7'}"
>
	{text}
</div>
