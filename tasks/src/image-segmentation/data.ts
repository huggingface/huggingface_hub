import type { TaskData } from "../Types";

import { PipelineType } from "../../../js/src/lib/interfaces/Types";
import { TASKS_MODEL_LIBRARIES } from "../const";

const taskData: TaskData = {
	datasets: [
		{
			description: "Widely used benchmark dataset for multiple Vision tasks.",
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
			description: "Average Precision (AP) is the Area Under the PR Curve (AUC-PR). It is calculated for each semantic class separately",
			id:          "Average Precision",
		},
		{
			description: "Mean Average Precision (mAP) is the overall average of the AP values",
			id:          "Mean Average Precision",
		},
		{
			description: "Intersection over Union (IoU) is the overlap of segmentation masks. Mean IoU is the average of the IoU of all semantic classes",
			id:          "Mean Intersection over Union",
		},
		{
			description: "APα is the Average Precision at the IoU threshold of a α value, for example, AP50 and AP75",
			id:          "APα",
		},
	],
	models: [
		{
			// TO DO: write description
			description: "Solid panoptic segmentation model trained on the COCO 2017 benchmark dataset.",
			id:          "facebook/detr-resnet-50-panoptic",
		}
	],
	summary:      "Image Segmentation divides an image into segments where each pixel in the image is mapped to an object. This task has multiple variants such as instance segmentation, panoptic segmentation and semantic segmentation.",
	widgetModels: ["facebook/detr-resnet-50-panoptic"],
	youtubeId:    "dKE8SIt9C-w",
};

export default taskData;
