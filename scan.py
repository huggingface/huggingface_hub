"""[TO BE DELETED]
Dump example to show what information is returned by `scan_cache_dir()`.

On my computer (luckily not much cache there):

```
(.venv310) ➜  huggingface_hub git:(972-utility-to-list-cache) python scan.py
Datasets:
  glue 113.6KiB (15 files)
    /Users/lucain/.cache/huggingface/hub/datasets--glue
    Revisions:
        9338f7b671827df886678df2bdd7cc7b4f36dffd: 2.4.0, main
          95.4KiB (14 files)
        f021ae41c879fcabcf823648ec685e3fead91fe7: 1.17.0
          95.5KiB (14 files)
  google--fleurs 61.9MiB (6 files)
    /Users/lucain/.cache/huggingface/hub/datasets--google--fleurs
    Revisions:
        129b6e96cf1967cd5d2b9b6aec75ce6cce7c89e8: refs/pr/1
          24.8KiB (3 files)
        24f85a01eb955224ca3946e70050869c56446805: main
          61.8MiB (4 files)

Models:
  bert-base-cased 1.8GiB (13 files)
    /Users/lucain/.cache/huggingface/hub/models--bert-base-cased
    Revisions:
        a8d257ba9925ef39f3036bfc338acf5283c512d9: main
          1.3GiB (9 files)
        378aa1bda6387fd00e824948ebe3488630ad8565: (detached)
          1.4GiB (9 files)
  t5-small 925.8MiB (11 files)
    /Users/lucain/.cache/huggingface/hub/models--t5-small
    Revisions:
        98ffebbb27340ec1b1abd7c45da12c253ee1882a: refs/pr/1
          692.6MiB (6 files)
        d78aea13fa7ecd06c29e3e46195d6341255065d5: main
          925.8MiB (9 files)
        d0a119eedb3718e34c648e594394474cf95e0617: (detached)
          463.3MiB (6 files)
```

"""
from huggingface_hub.utils import (
    CachedRepoInfo,
    CachedRevisionInfo,
    scan_cache_dir,
)


hf_cache_info = scan_cache_dir()


def print_revision(revision: CachedRevisionInfo) -> None:
    name = "(detached)" if revision.refs is None else ", ".join(sorted(revision.refs))
    print("      " + revision.commit_hash + ": " + name)
    print(
        "        "
        + revision.size_on_disk_str
        + " ("
        + str(revision.nb_files)
        + " files)"
    )


def print_cached_repo_info(cached_repo: CachedRepoInfo) -> None:
    print(
        cached_repo.repo_type.capitalize()
        + " "
        + cached_repo.repo_id
        + " "
        + cached_repo.size_on_disk_str
        + " ("
        + str(cached_repo.nb_files)
        + " files)"
    )
    print("  " + str(cached_repo.repo_path))
    print("  Revisions:")

    referenced = [
        revision for revision in cached_repo.revisions if revision.refs is not None
    ]
    for revision in referenced:
        print_revision(revision)

    detached = [revision for revision in cached_repo.revisions if revision.refs is None]
    for revision in detached:
        print_revision(revision)

for cached_repo in hf_cache_info.repos:
    print_cached_repo_info(cached_repo)
