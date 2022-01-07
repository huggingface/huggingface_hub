# Searching the Hub Efficiently with Python

In this tutorial, we will explore how to interact and explore the Hugging Face Hub with the `huggingface_hub` library to help find available models and datasets quickly.

## The Basics

The `huggingface_hub` comes packaged with an interface that can interact with the Hub in the `HfApi` class:


```python
>>> from huggingface_hub import HfApi
>>> api = HfApi()
```

This class let's you perform a variety of operations that interact with the raw Hub API. We'll be focusing on two specific funtions:
- `list_models`
- `list_datasets`

If you look at what can be passed into each function, you will find the parameter list looks something like:
- `filter`
- `author`
- `search`
- ...

Two of these parameters are intuitive (`author` and `search`), but what about that `filter`? 🤔 Let's dive into a few helpers quickly and revisit that question.

## Search Parameters

The `huggingface_hub` provides a user-friendly interface to know what exactly can be passed into this `filter` parameter through the `ModelSearchArguments` and `DatasetSearchArguments` classes:


```python
>>> from huggingface_hub import ModelSearchArguments, DatasetSearchArguments

>>> model_args = ModelSearchArguments()
>>> dataset_args = DatasetSearchArguments()
```

These are nested namespace objects that have **every single option** available on the Hub and that will return what should be passed to `filter`. The best of all is: it has tab completion 🎊 .

Let's pose a problem that would be complicated to solve without access to this information:
> I want to search the Hub for all PyTorch models trained on the `glue` dataset that can do Text Classification.

If you check what is available in `model_args` by checking it's output, you will find:


```python
>>> model_args
```




    Available Attributes or Keys:
     * author
     * dataset
     * language
     * library
     * license
     * model_name
     * pipeline_tag



It has a variety of attributes or keys available to you. This is because it is both an object and a dictionary, so you can either do `model_args["author"]` or `model_args.author`. For this tutorial, let's follow the latter format.

The first criteria is getting all PyTorch models. This would be found under the `library` attribute, so let's see if it is there:


```python
>>> model_args.library
```




    Available Attributes or Keys:
     * AdapterTransformers
     * Asteroid
     * ESPnet
     * Flair
     * JAX
     * Joblib
     * Keras
     * ONNX
     * PyTorch
     * Pyannote
     * Rust
     * Scikit_learn
     * SentenceTransformers
     * Stanza
     * TFLite
     * TensorBoard
     * TensorFlow
     * TensorFlowTTS
     * Timm
     * Transformers
     * allennlp
     * fastText
     * fastai
     * spaCy
     * speechbrain



It is! The `PyTorch` name is there, so you'll need to use `model_args.library.PyTorch`:


```python
>>> model_args.library.PyTorch
```




    'pytorch'



Below is an animation repeating the process for finding both the `Text Classification` and `glue` requirements:

![](../assets/hub/search_text_classification.gif)

![](../assets/hub/search_glue.gif)

Now that all the pieces are there, the last step is to combine them all for something the API can use through the `ModelFilter` and `DatasetFilter` classes. The classes transform the outputs of the previous step into something the API can use conveniently:


```python
>>> from huggingface_hub import ModelFilter, DatasetFilter

>>> filt = ModelFilter(
>>>     task=args.pipeline_tag.TextClassification, 
>>>     trained_dataset=args.dataset.glue, 
>>>     library=args.library.PyTorch
>>> )
>>> api.list_models(filter=filt)[0]
```




    ModelInfo: {
    	modelId: 09panesara/distilbert-base-uncased-finetuned-cola
    	sha: f89a85cb8703676115912fffa55842f23eb981ab
    	lastModified: 2021-12-21T14:03:01.000Z
    	tags: ['pytorch', 'tensorboard', 'distilbert', 'text-classification', 'dataset:glue', 'transformers', 'license:apache-2.0', 'generated_from_trainer', 'model-index', 'infinity_compatible']
    	pipeline_tag: text-classification
    	siblings: [ModelFile(rfilename='.gitattributes'), ModelFile(rfilename='.gitignore'), ModelFile(rfilename='README.md'), ModelFile(rfilename='config.json'), ModelFile(rfilename='pytorch_model.bin'), ModelFile(rfilename='special_tokens_map.json'), ModelFile(rfilename='tokenizer.json'), ModelFile(rfilename='tokenizer_config.json'), ModelFile(rfilename='training_args.bin'), ModelFile(rfilename='vocab.txt'), ModelFile(rfilename='runs/Dec21_13-51-40_bc62d5d57d92/events.out.tfevents.1640094759.bc62d5d57d92.77.0'), ModelFile(rfilename='runs/Dec21_13-51-40_bc62d5d57d92/events.out.tfevents.1640095117.bc62d5d57d92.77.2'), ModelFile(rfilename='runs/Dec21_13-51-40_bc62d5d57d92/1640094759.4067502/events.out.tfevents.1640094759.bc62d5d57d92.77.1')]
    	config: None
    	private: False
    	downloads: 6
    	library_name: transformers
    	likes: 0
    }



As we can see, it found the models we wanted that fit all of our criteria. We can even take it further by passing in an array for each of the parameters we had before. For example, now we'll look for the same configuration, but also include `TensorFlow` in the filter:


```python
>>> filt = ModelFilter(
>>>     task=args.pipeline_tag.TextClassification, 
>>>     library=[args.library.PyTorch, args.library.TensorFlow]
>>> )
>>> api.list_models(filter=filt)[0]
```




    ModelInfo: {
    	modelId: CAMeL-Lab/bert-base-arabic-camelbert-ca-poetry
    	sha: bc50b6dc1c97dc66998287efb6d044bdaa8f7057
    	lastModified: 2021-10-17T12:09:38.000Z
    	tags: ['pytorch', 'tf', 'bert', 'text-classification', 'ar', 'arxiv:1905.05700', 'arxiv:2103.06678', 'transformers', 'license:apache-2.0', 'infinity_compatible']
    	pipeline_tag: text-classification
    	siblings: [ModelFile(rfilename='.gitattributes'), ModelFile(rfilename='README.md'), ModelFile(rfilename='config.json'), ModelFile(rfilename='pytorch_model.bin'), ModelFile(rfilename='special_tokens_map.json'), ModelFile(rfilename='tf_model.h5'), ModelFile(rfilename='tokenizer_config.json'), ModelFile(rfilename='training_args.bin'), ModelFile(rfilename='vocab.txt')]
    	config: None
    	private: False
    	downloads: 21
    	library_name: transformers
    	likes: 0
    }



With these two functionalities combined, you can search for all available parameters and tags within the Hub to search for with ease for both Datasets and Models
