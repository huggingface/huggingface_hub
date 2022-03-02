<script lang="ts">
	import type { ModelData } from "../../interfaces/Types";
	import IconCheckmark from "../Icons/IconCheckmark.svelte";
	import IconInfo from "../Icons/IconInfo.svelte";

	export let model: ModelData;
	const modelIndex = model["model-index"];
	const pwcLink = model.pwcLink;
	const verified = false; // not implemented yet
</script>

{#if modelIndex}
	<div class="divider-column-vertical" />
	<h2
		class="mb-5 font-semibold text-gray-800 flex items-center whitespace-nowrap overflow-hidden text-smd"
	>
		<svg
			xmlns="http://www.w3.org/2000/svg"
			xmlns:xlink="http://www.w3.org/1999/xlink"
			aria-hidden="true"
			role="img"
			class="flex-none mr-2 w-3 text-gray-300"
			width="1em"
			height="1em"
			preserveAspectRatio="xMidYMid meet"
			viewBox="0 0 32 32"
		>
			<path d="M30 30h-8V4h8z" fill="currentColor" />
			<path d="M20 30h-8V12h8z" fill="currentColor" />
			<path d="M10 30H2V18h8z" fill="currentColor" />
		</svg>
		Evaluation results
		<a
			target="_blank"
			href="https://github.com/huggingface/huggingface_hub/blame/main/modelcard.md"
		>
			<IconInfo
				classNames="ml-1.5 text-sm text-gray-400 hover:text-black dark:hover:text-white"
			/>
		</a>
	</h2>
	{#if "error" in modelIndex}
		<div class="border border-gray-100 rounded-lg p-4 mb-6">
			<h4 class="text-sm">Model card error</h4>
			<p class="text-sm text-grey mt-2">
				This model's model-index metadata is invalid:
				{modelIndex.error}
			</p>
		</div>
	{:else}
		<ul class="space-y-1.5 text-sm">
			<!-- from here on, valid model index set -->
			{#each modelIndex.map((x) => x.results).flat() as result}
				{#each result.metrics as metric}
					<li class="flex items-center hover:bg-gray-50 dark:hover:bg-gray-900">
						<div
							class="mr-2 truncate"
							title="{result.task.name ?? result.task.type}: {metric.name ??
								metric.type} on {result.dataset
								? result.dataset.name ?? result.dataset.type
								: 'Unknown dataset'}"
						>
							{metric.name ?? metric.type}
							{#if result.dataset}
								on {result.dataset.name ?? result.dataset.type}
							{/if}
						</div>
						{#if verified}
							<a
								target="_blank"
								class="leading-snug bg-white border border-green-500 text-xs text-green-500 rounded px-1 flex flex-shrink-0 items-center"
								href={`/${model.id}/blob/${model.branch}/README.md`}
							>
								<IconCheckmark classNames="w-3 flex-none mr-0.5" />
								verified
							</a>
						{:else}
							<a
								target="_blank"
								class="leading-snug bg-white border border-gray-400 text-xs text-gray-400 rounded px-1 flex flex-shrink-0 items-center"
								href={`/${model.id}/blob/${model.branch}/README.md`}
							>
								self-reported
							</a>
						{/if}
						<div class="border-b border-dashed flex-1 mx-3 self-end mb-1.5" />
						<div class="font-mono text-xs ml-auto">
							{typeof metric.value === "number"
								? metric.value.toFixed(3)
								: metric.value}
						</div>
					</li>
				{/each}
			{/each}
		</ul>
	{/if}

	{#if !("error" in modelIndex)}
		<div href="#" class="mt-auto pt-3 flex items-center text-xs text-gray-500">
			<a
				target="_blank"
				href={pwcLink && "url" in pwcLink
					? pwcLink.url
					: "https://github.com/huggingface/huggingface_hub/blame/main/modelcard.md"}
				class="flex items-center hover:underline"
			>
				<svg
					xmlns="http://www.w3.org/2000/svg"
					xmlns:xlink="http://www.w3.org/1999/xlink"
					aria-hidden="true"
					role="img"
					class="iconify iconify--carbon mr-1"
					width="1em"
					height="1em"
					preserveAspectRatio="xMidYMid meet"
					viewBox="0 0 32 32"
				>
					<circle cx="7" cy="9" r="3" fill="currentColor" />
					<circle cx="7" cy="23" r="3" fill="currentColor" />
					<path d="M16 22h14v2H16z" fill="currentColor" />
					<path d="M16 8h14v2H16z" fill="currentColor" />
				</svg>
				{#if pwcLink && "error" in pwcLink}
					{pwcLink.error}
				{:else}
					View leaderboard (Papers With Code)
				{/if}
			</a>
		</div>
	{/if}
{/if}
