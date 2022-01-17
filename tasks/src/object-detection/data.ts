import type { TaskData } from "../Types";

import { PipelineType } from "../../../js/src/lib/interfaces/Types";
import { TASKS_MODEL_LIBRARIES } from "../const";

const taskData: TaskData = {
	datasets: [
		{
			// TODO write proper description
			description: "Widely used benchmark dataset for multiple Vision tasks.",
			id:          "merve/coco2017",
		},
	],
	demo: {
		inputs: [
			{
				filename: "object-detection-input.jpg",
				type:     "img",
			},
		],
		outputs: [
			{
				filename: "object-detection-output.jpg",
				type:     "img",
			},
		],
	},
	id:        "object-detection",
	label:     PipelineType["object-detection"],
	libraries: TASKS_MODEL_LIBRARIES["object-detection"],
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
			description: "The APα metric is the Average Precision at the IoU threshold of a α value, for example, AP50 and AP75",
			id:          "APα",
		},
	],
	models: [
		{
			// TO DO: write description
			description: "Solid object detection model trained on the benchmark dataset COCO 2017.",
			id:          "facebook/detr-resnet-50",
		}
	],
	summary:      "Object Detection models allow users to identify objects of certain defined classes. Object detection models receive an image as input and output the images with bounding boxes and labels on detected objects.",
	widgetModels: ["facebook/detr-resnet-50"],
	youtubeId:    "WdAeKSOpxhw",
};

export default taskData;
