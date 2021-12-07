import type { TaskData } from "../Types";

import { PipelineType } from "../../../widgets/src/lib/interfaces/Types";
import { TASKS_MODEL_LIBRARIES } from "../const";

const taskData: TaskData = {
	datasets: [
		{
			// TODO write proper description
			description: "Benchmark dataset used for image classification, with images that belong to 100 classes",
			id:          "cifar100",
		},
		{
			// TODO write proper description
			description: "Dataset consisting of images of garments",
			id:          "fashion-mnist",
		},
	],
	demo: {
		inputs: [
			{
				filename: "image-classification-input.jpg",
				type:     "img",
			},
		],
		outputs: [
			{
				filename: "A JSON response including class labels",
				type:     "img",
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
			id:          "f1-Score",
		},

	],
	models: [
		{
			// TO DO: write description
			description: "Strong image classification model trained on ImageNet dataset",
			id:          "google/vit-base-patch16-224",
		},
		{
			// TO DO: write description
			description: "Strong image classification model trained on ImageNet dataset",
			id:          "facebook/deit-base-distilled-patch16-224",
		},
	],
	summary:      "Image classification is the task of assigning a class to an entire image. The images are expected to have only one class instance in one image. Image classification models take an image as input and return class labels.",
	widgetModels: ["google/vit-base-patch16-224"],
	youtubeId:    "",
};

export default taskData;
