<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Discussions और Pull Requests के साथ इंटरैक्ट करें

`huggingface_hub` library, Hub पर Pull Requests और Discussions के साथ इंटरैक्ट करने के लिए एक Python interface प्रदान करती है।
Hub पर Discussions और Pull Requests क्या हैं और वे अंदरूनी तौर पर कैसे काम करते हैं, इसकी गहरी समझ के लिए
[समर्पित documentation पेज](https://huggingface.co/docs/hub/repositories-pull-requests-discussions) देखें।

## Hub से Discussions और Pull Requests प्राप्त करें

`HfApi` class आपको किसी दिए गए repo पर Discussions और Pull Requests प्राप्त करने की सुविधा देती है:

```python
>>> from huggingface_hub import get_repo_discussions
>>> for discussion in get_repo_discussions(repo_id="bigscience/bloom"):
...     print(f"{discussion.num} - {discussion.title}, pr: {discussion.is_pull_request}")

# 11 - Add Flax weights, pr: True
# 10 - Update README.md, pr: True
# 9 - Training languages in the model card, pr: True
# 8 - Update tokenizer_config.json, pr: True
# 7 - Slurm training script, pr: False
[...]
```

`HfApi.get_repo_discussions`, author, type (Pull Request या Discussion) और status (`open` या `closed`) के आधार पर filtering का समर्थन करता है:

```python
>>> from huggingface_hub import get_repo_discussions
>>> for discussion in get_repo_discussions(
...    repo_id="bigscience/bloom",
...    author="ArthurZ",
...    discussion_type="pull_request",
...    discussion_status="open",
... ):
...     print(f"{discussion.num} - {discussion.title} by {discussion.author}, pr: {discussion.is_pull_request}")

# 19 - Add Flax weights by ArthurZ, pr: True
```

`HfApi.get_repo_discussions` एक [generator](https://docs.python.org/3.7/howto/functional.html#generators) return करता है जो
[`Discussion`] objects yield करता है। सभी Discussions को एक ही list में पाने के लिए, चलाएँ:

```python
>>> from huggingface_hub import get_repo_discussions
>>> discussions_list = list(get_repo_discussions(repo_id="bert-base-uncased"))
```

[`HfApi.get_repo_discussions`] द्वारा return किया गया [`Discussion`] object, Discussion या Pull Request का high-level overview रखता है।
आप [`HfApi.get_discussion_details`] का उपयोग करके अधिक विस्तृत जानकारी भी प्राप्त कर सकते हैं:

```python
>>> from huggingface_hub import get_discussion_details

>>> get_discussion_details(
...     repo_id="bigscience/bloom-1b3",
...     discussion_num=2
... )
DiscussionWithDetails(
    num=2,
    author='cakiki',
    title='Update VRAM memory for the V100s',
    status='open',
    is_pull_request=True,
    events=[
        DiscussionComment(type='comment', author='cakiki', ...),
        DiscussionCommit(type='commit', author='cakiki', summary='Update VRAM memory for the V100s', oid='1256f9d9a33fa8887e1c1bf0e09b4713da96773a', ...),
    ],
    conflicting_files=[],
    target_branch='refs/heads/main',
    merge_commit_oid=None,
    diff='diff --git a/README.md b/README.md\nindex a6ae3b9294edf8d0eda0d67c7780a10241242a7e..3a1814f212bc3f0d3cc8f74bdbd316de4ae7b9e3 100644\n--- a/README.md\n+++ b/README.md\n@@ -132,7 +132,7 [...]',
)
```

[`HfApi.get_discussion_details`] एक [`DiscussionWithDetails`] object return करता है, जो [`Discussion`] का subclass है और
Discussion या Pull Request के बारे में अधिक विस्तृत जानकारी रखता है। इस जानकारी में [`DiscussionWithDetails.events`] के माध्यम से
Discussion के सभी comments, status changes और renames शामिल होते हैं।

Pull Request की स्थिति में, आप [`DiscussionWithDetails.diff`] के साथ raw git diff प्राप्त कर सकते हैं। Pull Request के सभी
commits [`DiscussionWithDetails.events`] में सूचीबद्ध होते हैं।


## किसी Discussion या Pull Request को programmatically बनाएँ और edit करें

[`HfApi`] class, Discussions और Pull Requests को बनाने और edit करने के तरीके भी प्रदान करती है।
Discussions या Pull Requests बनाने और edit करने के लिए आपको एक [access token](https://huggingface.co/docs/hub/security-tokens) की आवश्यकता होगी।

Hub पर किसी repo में बदलाव प्रस्तावित करने का सबसे आसान तरीका [`create_commit`] API के माध्यम से है: बस
`create_pr` parameter को `True` सेट करें। यह parameter उन अन्य methods पर भी उपलब्ध है जो [`create_commit`] को wrap करते हैं:

    * [`upload_file`]
    * [`upload_folder`]
    * [`delete_file`]
    * [`delete_folder`]
    * [`metadata_update`]

```python
>>> from huggingface_hub import metadata_update

>>> metadata_update(
...     repo_id="username/repo_name",
...     metadata={"tags": ["computer-vision", "awesome-model"]},
...     create_pr=True,
... )
```

आप किसी repo पर Discussion बनाने के लिए [`HfApi.create_discussion`] (और इसी तरह Pull Request बनाने के लिए [`HfApi.create_pull_request`]) का भी उपयोग कर सकते हैं।
इस तरह Pull Request खोलना तब उपयोगी हो सकता है जब आपको बदलावों पर locally काम करना हो। इस तरह खोले गए Pull Requests `"draft"` mode में होंगे।

```python
>>> from huggingface_hub import create_discussion, create_pull_request

>>> create_discussion(
...     repo_id="username/repo-name",
...     title="Hi from the huggingface_hub library!",
...     token="<insert your access token here>",
... )
DiscussionWithDetails(...)

>>> create_pull_request(
...     repo_id="username/repo-name",
...     title="Hi from the huggingface_hub library!",
...     token="<insert your access token here>",
... )
DiscussionWithDetails(..., is_pull_request=True)
```

Pull Requests और Discussions का प्रबंधन पूरी तरह [`HfApi`] class के साथ किया जा सकता है। उदाहरण के लिए:

    * [`comment_discussion`] — comments जोड़ने के लिए
    * [`edit_discussion_comment`] — comments edit करने के लिए
    * [`rename_discussion`] — किसी Discussion या Pull Request का नाम बदलने के लिए
    * [`change_discussion_status`] — किसी Discussion / Pull Request को खोलने या बंद करने के लिए
    * [`merge_pull_request`] — किसी Pull Request को merge करने के लिए


सभी उपलब्ध methods के विस्तृत reference के लिए [`HfApi`] documentation पेज देखें।

## CLI से Discussions और Pull Requests का प्रबंधन करें

ऊपर दिए गए सभी operations command line से `hf discussions` के माध्यम से भी उपलब्ध हैं। यह scripting, CI pipelines,
या Python code लिखे बिना त्वरित इंटरैक्शन के लिए उपयोगी है।

```bash
# किसी repo पर खुली discussions और PRs सूचीबद्ध करें
hf discussions list bigscience/bloom

# किसी dataset repo पर discussions सूचीबद्ध करें
hf discussions list nebius/SWE-rebench-V2 --type dataset

# comments सहित किसी विशिष्ट discussion की जानकारी प्राप्त करें
hf discussions info bigscience/bloom 2 --comments

# एक नई discussion बनाएँ
hf discussions create username/repo-name --title "Bug report" --body "Description here"

# एक pull request बनाएँ
hf discussions create username/repo-name --title "Fix typo" --pull-request

# किसी discussion या PR पर comment करें
hf discussions comment username/repo-name 5 --body "LGTM!"

# किसी pull request को merge करें
hf discussions merge username/repo-name 5 --yes

# किसी pull request का diff दिखाएँ
hf discussions diff username/repo-name 5
```

options की पूरी सूची के लिए, `hf discussions --help` चलाएँ या [CLI reference](./cli#hf-discussions) देखें।

## किसी Pull Request में बदलाव push करें

*जल्द ही आ रहा है !*

## यह भी देखें

अधिक विस्तृत reference के लिए, [Discussions and Pull Requests](../package_reference/community) और
[hf_api](../package_reference/hf_api) documentation पेज देखें।
