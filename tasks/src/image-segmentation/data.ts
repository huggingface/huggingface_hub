import type { TaskData } from "../Types";

import { PipelineType } from "../../../widgets/src/lib/interfaces/Types";
import { TASKS_MODEL_LIBRARIES } from "../const";

const taskData: TaskData = {
	datasets: [
		{
			description: "The COCO dataset is widely used as a benchmark for models used for image segmentation, object detection, and captioning",
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
			description: "The Average Precision (AP) metric is the Area Under the PR Curve (AUC-PR). It is calculated for each class separately",
			id:          "Average Precision",
		},
		{
			description: "The Mean Average Precision (mAP) metric is the overall average of the AP values",
			id:          "Mean Average Precision",
		},
		{
			description: "The Intersection over Union (IoU) metric is the overlap of segmentation masks. Mean IoU is the average of the IoU of all semantic classes",
			id:          "Mean Intersection over Union",
		},
		{
			description: "The APα metric is the Average Precision at the IoU threshold of a α value, for example, AP50 and AP75",
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
	summary:      "Image Segmentation divides an image into segments where each pixel in the image is mapped to an object. This task has multiple variants such as instance segmentation, panoptic segmentation and semantic segmentation",
	widgetModels: ["facebook/detr-resnet-50"],
	youtubeId:    "",
};

export default taskData;
