import datasets


_CITATION = """\
"""

_DESCRIPTION = """\
This is a test dataset.
"""

_URLS = {"train": "https://pastebin.com/raw/HvpE1CnA", "dev": "some_text.txt"}


class Test(datasets.GeneratorBasedBuilder):
    """SQUAD: The Stanford Question Answering Dataset. Version 1.1."""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://huggingface.co/datasets/lhoestq/test",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files["dev"]},
            ),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        for _id, line in enumerate(open(filepath, encoding="utf-8")):
            yield _id, {"text": line.rstrip()}
