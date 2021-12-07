import type { TaskData } from "../Types";

import { PipelineType } from "../../../widgets/src/lib/interfaces/Types";
import { TASKS_MODEL_LIBRARIES } from "../const";

const taskData: TaskData = {
	datasets: [
		{
			// TODO write proper description
			description: "Benchmark dataset used for object detection",
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
			description: "Strong object detection model trained on benchmark dataset COCO 2017.",
			id:          "facebook/detr-resnet-50",
		},
		{
			// TO DO: write description
			description: "Strong object detection model trained on benchmark dataset COCO 2017.",
			id:          "facebook/detr-resnet-101",
		},
	],
	summary:      "Object detection is a computer vision task. Models trained on object detection task allows users to detect instances of objects of classes given an image. Object detection models receive an image as an input and output the images including bounding boxes and labels on the detected objects.",
	widgetModels: ["facebook/detr-resnet-50"],
	youtubeId:    "",
};

export default taskData;
