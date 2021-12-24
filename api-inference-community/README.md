
This repositories enable third-party libraries integrated with [huggingface_hub](https://github.com/huggingface/huggingface_hub/) to create
their own docker so that the widgets on the hub can work as the `transformers` one do.

The hardware to run the API will be provided by Hugging Face for now.

The `docker_images/common` folder is intended to be a starter point for all new libs that 
want to be integrated.

### Adding a new container from a new lib.


1. Copy the `docker_images/common` folder into your library's name `docker_images/example`.
2. Edit:
    - `docker_images/example/requirements.txt`
    - `docker_images/example/app/main.py`
    - `docker_images/example/app/pipelines/{task_name}.py` 
    to implement the desired functionnality. All required code is marked with `IMPLEMENT_THIS` markup.
3. Feel free to customize anything required by your lib everywhere you want. The only real requirements, are to honor the HTTP endpoints, in the same fashion as the `common` folder for all your supported tasks.
4. Edit `example/tests/test_api.py` to add TESTABLE_MODELS.
5. Pass the test suite `pytest -sv --rootdir docker_images/example/ docker_images/example/`
6. Submit your PR and enjoy !

### Going the full way

Doing the first 6 steps is good enough to get started, however in the process 
you can anticipate some problems corrections early on. Maintainers will help you
along the way if you don't feel confident to follow those steps yourself

1. Test your creation within a docker

```python
./manage.py docker --model-id MY_MODEL
```

should work and responds on port 8000. `curl -X POST -d "test" http://localhost:8000` for instance if 
the pipeline deals with simple text.

If it doesn't work out of the box and/or docker is slow for some reason you
can test locally (using your local python environment) with :

`./manage.py start --model-id MY_MODEL`


2. Test your docker uses cache properly.

When doing subsequent docker launch with the same model_id, the docker should start up very fast and not redownload the whole model file. If you see the model/repo being downloaded over and over, it means the cache is not being used correctly.
You can edit the `docker_images/{framework}/Dockerfile` and add an environement variable (by default it assumes `HUGGINGFACE_HUB_CACHE`), or your code directly to put
the model files in the `/data` folder.

3. Add a docker test.

Edit the `tests/test_dockers.py` file to add a new test with your new framework
in it (`def test_{framework}(self):` for instance). As a basic you should have 1 line per task in this test function with a real working model on the hub. Those tests are relatively slow but will check automatically that correct errors are replied by your API and that the cache works properly. To run those tests your can simply do:

```bash

RUN_DOCKER_TESTS=1 pytest -sv tests/test_dockers.py::DockerImageTests::test_{framework}
```

### Modifying files within `api-inference-community/{routes,validation,..}.py`.

If you ever come across a bug within `api-inference-community/` package or want to update it
the developpement process is slightly more involved.

- First, make sure you need to change this package, each framework is very autonomous
 so if your code can get away by being standalone go that way first as it's much simpler.
- If you can make the change only in `api-inference-community` without depending on it
that's also a great option. Make sure to add the proper tests to your PR.
- Finally, the best way to go is to develop locally using `manage.py` command:
- Do the necessary modifications within `api-inference-community` first.
- Install it locally in your environment with `pip install -e .`
- Install your package dependencies locally.
- Run your webserver locally: `./manage.py start --framework example --task audio-source-separation --model-id MY_MODEL`
- When everything is working, you will need to split your PR in two, 1 for the `api-inference-community` part.
  The second one will be for your package specific modifications and will only land once the `api-inference-community`
  tag has landed.
- This workflow is still work in progress, don't hesitate to ask questions to maintainers.

Another similar command `./manage.py docker --framework example --task audio-source-separation --model-id MY_MODEL`
Will launch the server, but this time in a protected, controlled docker environment making sure the behavior
will be exactly the one in the API.



### Available tasks

- **Automatic speech recognition**: Input is a file, output is a dict of understood words being said within the file
- **Text generation**: Input is a text, output is a dict of generated text
- **Image recognition**: Input is an image, output is a dict of generated text
- **Question answering**: Input is a question + some context, output is a dict containing necessary information to locate the answer to the `question` within the `context`.
- **Audio source separation**: Input is some audio, and the output is n audio files that sum up to the original audio but contain individual soures of sound (either speakers or instruments for instant).
- **Token classification**: Input is some text, and the output is a list of entities mentionned in the text. Entities can be anything remarquable like locations, organisations, persons, times etc...
- **Text to speech**: Input is some text, and the output is an audio file saying the text...
- **Sentence Similarity**: Input is some sentence and a list of reference sentences, and the list of similarity scores.

