---
title: Adding a task
---

# Adding a task to the Hub

## What's a task?

Tasks, or pipeline types, describe the "shape" of each model's API (inputs and outputs) and are used to determine which Inference API and widget we want to display for any given model. 

![/docs/assets/hub/tasks.png](/docs/assets/hub/tasks.png)

This classification is rather coarse-grained (you can always add more fine-grained task name in your model tags), so **you should rarely have to create a new task**. If you do want to add support for a new task, this document explains the required steps.

## Overview

Having a new task fully integrated in the Hub means that:
* Users can search for all models of a given task.
* The Inference API supports the task.
* Users can try out models directly with the widget. 🏆

Note that you're not expected to implement all the steps. Adding a new task is a community effort, and multiple people can contribute to it. 🧑‍🤝‍🧑

To begin the process, open a new issue in the [huggingface_hub](https://github.com/huggingface/huggingface_hub/issues) repository. Please use the "Adding a new task" template. ⚠️Before doing any coding, it's suggested to go over this document. ⚠️

The first step is to upload a model for your proposed task. Once you have a model in the Hub for the new task, the next step is to enable it in the Inference API. There are three types of support that you can choose from:

* 🤗 using a `transformers` model
* 🐳 using a model from an [officially supported library](/docs/hub/libraries)
* 🖨️ using a model with custom inference code. This option is experimental and has downsides, so it's always suggested to use any of the other two approaches if possible.

Finally, there are a couple of UI elements that can be added, such as the task icon and the widget, that complete the integration in the Hub. 📷 

Some steps are orthogonal, you don't need to do them in order. **You don't need the Inference API to add the icon.** This means that, even if there isn't full integration at the moment, users can still search for models of a given task.

## Using Hugging Face transformers library

If your model is a `transformers`-based model, there is a 1:1 mapping between the Inference API task and a `pipeline` class. Here are some example PRs from the `transformers` library:
* [Adding ImageClassificationPipeline](https://github.com/huggingface/transformers/pull/11598)
* [Adding AudioClassificationPipeline](https://github.com/huggingface/transformers/pull/13342)

Once the pipeline is submitted and deployed, you should be able to use the Inference API for your model.

## Using Community Inference API with a supported library

The Hub also supports over 10 open-source libraries in the [Community Inference API](https://github.com/huggingface/huggingface_hub/tree/main/api-inference-community). 

**Adding a new task is relatively straightforward and requires 2 PRs:**
* PR 1: Add the new task to the API [validation](https://github.com/huggingface/huggingface_hub/blob/main/api-inference-community/api_inference_community/validation.py). This code ensures that the inference input is valid for a given task. Some PR examples:
    * [Add text-to-image](https://github.com/huggingface/huggingface_hub/commit/5f040a117cf2a44d704621012eb41c01b103cfca#diff-db8bbac95c077540d79900384cfd524d451e629275cbb5de7a31fc1cd5d6c189)
    * [Add audio-classification](https://github.com/huggingface/huggingface_hub/commit/141e30588a2031d4d5798eaa2c1250d1d1b75905#diff-db8bbac95c077540d79900384cfd524d451e629275cbb5de7a31fc1cd5d6c189)
    * [Add structured-data-classification](https://github.com/huggingface/huggingface_hub/commit/dbea604a45df163d3f0b4b1d897e4b0fb951c650#diff-db8bbac95c077540d79900384cfd524d451e629275cbb5de7a31fc1cd5d6c189)
* PR 2: Add the new task to a library docker image. You should also add a template to [`docker_images/common/app/pipelines`](https://github.com/huggingface/huggingface_hub/tree/main/api-inference-community/docker_images/common/app/pipelines) to facilitate integrating the task in other libraries. Here is an example PR:
    * [Add text-classification to spaCy](https://github.com/huggingface/huggingface_hub/commit/6926fd9bec23cb963ce3f58ec53496083997f0fa#diff-3f1083a92ca0047b50f9ad2d04f0fe8dfaeee0e26ab71eb8835e365359a1d0dc)

## Adding Community Inference API for a quick prototype

**My model is not supported by any library. Am I doomed? 😱**

No, you're not! We have an experimental Docker image for quick prototyping of new tasks and introduction of new libraries. This is the [generic Inference API](https://github.com/huggingface/huggingface_hub/tree/main/api-inference-community/docker_images/generic) and should allow you to have a new task in production with very little development from your side.

How does it work from the user point of view? Users create a copy of a [template](https://huggingface.co/templates) repo for their given task. Users then need to define their `requirements.txt` and fill `pipeline.py`. Note that this is not intended for fast production use cases, but more for quick experimentation and prototyping.


## UI elements

The Hub allows users to filter models by a given task. In order to do this, you need to add the task to a couple of places and pick an icon for the task.

1. Add the task type to `Types.ts`

In [interfaces/Types.ts](https://github.com/huggingface/huggingface_hub/blob/main/js/src/lib/interfaces/Types.ts), you need to do a couple of things

* Add the type to `PipelineType`. Note that they are sorted in different categories (NLP, Audio, Computer Vision and others).
* Specify the task color in `PIPELINE_COLOR`. 
* Specify the display order in `PIPELINE_TAGS_DISPLAY_ORDER`.

2. Choose an icon

You can add an icon in the [lib/Icons](https://github.com/huggingface/huggingface_hub/tree/main/js/src/lib/Icons) directory. We usually choose carbon icons from https://icones.js.org/collection/carbon. 


## Widget

Once the task is in production, what could be more exciting that implementing some way for users to play directly with the models in their browser? 🤩 You can find all the widgets [here](https://huggingface-widgets.netlify.app/). 

In case you would be interested in contributing with a widget, you can look at the [implementation](https://github.com/huggingface/huggingface_hub/tree/main/js/src/lib/components/InferenceWidget/widgets) of all the widgets. You can also find WIP documentation on how to implement a widget in https://github.com/huggingface/huggingface_hub/tree/main/js. 