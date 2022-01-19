import type { TaskData } from "../Types";

import { PipelineType } from "../../../js/src/lib/interfaces/Types";
import { TASKS_MODEL_LIBRARIES } from "../const";

const taskData: TaskData = {
	datasets: [
		{
			// TODO write proper description
			description: "Benchmark dataset used for image classification with images that belong to 100 classes.",
			id:          "cifar100",
		},
		{
			// TODO write proper description
			description: "Dataset consisting of images of garments.",
			id:          "fashion-mnist",
		},
	],
	demo: {
		inputs: [
			{
				filename: "image-classification-input.jpeg",
				type:     "img",
			},
		],
		outputs: [
			{
				"type": "chart",
				data: [
					{
						"label": "Egyptian cat",
						"score": 0.514
					},
					{
						"label": "Tabby cat",
						"score": 0.193
					},
					{
						"label": "Tiger cat",
						"score": 0.068
					},
				]
			},
		],
	},
	id:        "image-classification",
	label:     PipelineType["image-classification"],
	libraries: TASKS_MODEL_LIBRARIES["image-classification"],
	metrics:   [
		{
			description: "",
			id:          "accuracy",
		},
		{
			description: "",
			id:          "recall",
		},
		{
			description: "",
			id:          "precision",
		},
		{
			description: "",
			id:          "f1",
		},

	],
	models: [
		{
			// TO DO: write description
			description: "Strong Image Classification model trained on the ImageNet dataset.",
			id:          "google/vit-base-patch16-224",
		},
		{
			// TO DO: write description
			description: "Strong Image Classification model trained on the ImageNet dataset.",
			id:          "facebook/deit-base-distilled-patch16-224",
		},
	],
	summary:      "Image classification is the task of assigning a label or class to an entire image. Images are expected to have only one class for each image.Image classification models take an image as input and return a prediction about which class the image belongs to.",
	widgetModels: ["google/vit-base-patch16-224"],
	youtubeId:    "tjAIM7BOYhw",
};

export default taskData;
