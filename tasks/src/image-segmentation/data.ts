import type { TaskData } from "../Types";

import { PipelineType } from "../../../widgets/src/lib/interfaces/Types";
import { TASKS_MODEL_LIBRARIES } from "../const";

const taskData: TaskData = {
	datasets: [
		{
			// TODO write proper description
			description: "Benchmark dataset used for the task",
			id:          "merve/coco2017",
		},
	],
	demo: {
		inputs: [
			{
				filename: "image-segmentation-input.jpeg",
				type:     "img",
			},
		],
		outputs: [
			{
				filename: "image-segmentation-output.png",
				type:     "img",
			},
		],
	},
	id:        "image-segmentation",
	label:     PipelineType["image-segmentation"],
	libraries: TASKS_MODEL_LIBRARIES["image-segmentation"],
	metrics:   [
		{
			description: "Average Precision (AP) is the Area Under the PR Curve (AUC-PR). It is calculated for each class separately.",
			id:          "Average Precision",
		},
		{
			description: "Mean Average Precision is the overall average of the Average Precision values.",
			id:          "Mean Average Precision",
		},
		{
			description: "APα is the Average Precision at IoU threshold of α, AP50 and AP75 are widely used.",
			id:          "APα",
		},
	],
	models: [
		{
			// TO DO: write description
			description: "Strong panoptic segmentation model trained on COCO 2017 benchmark dataset.",
			id:          "facebook/detr-resnet-50-panoptic",
		}
	],
	summary:      "Image segmentation task is to divide an image to segments, where every pixel in the image is assigned to an object. This task has multiple variants, instance segmentation, panoptic segmentation and semantic segmentation.",
	widgetModels: ["facebook/detr-resnet-50"],
	youtubeId:    "",
};

export default taskData;
