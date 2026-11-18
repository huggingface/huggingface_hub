# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from huggingface_hub.hf_api import PaperInfo


def test_paper_info_reads_metadata_nested_in_list_responses() -> None:
    paper = PaperInfo(
        paper={
            "id": "2611.16961",
            "githubRepo": "https://github.com/example/paper",
            "githubStars": 42,
            "projectPage": "https://example.org/project",
            "ai_summary": "A short summary",
            "ai_keywords": ["evaluation", "ranking"],
        },
        title="Example paper",
    )

    assert paper.id == "2611.16961"
    assert paper.github_repo == "https://github.com/example/paper"
    assert paper.github_stars == 42
    assert paper.project_page == "https://example.org/project"
    assert paper.ai_summary == "A short summary"
    assert paper.ai_keywords == ["evaluation", "ranking"]


def test_paper_info_keeps_top_level_metadata_for_paper_info_response() -> None:
    paper = PaperInfo(
        id="2611.16961",
        githubRepo="https://github.com/example/paper",
        githubStars=7,
        projectPage="https://example.org/project",
        ai_summary="A short summary",
        ai_keywords=["agents"],
    )

    assert paper.github_repo == "https://github.com/example/paper"
    assert paper.github_stars == 7
    assert paper.project_page == "https://example.org/project"
    assert paper.ai_summary == "A short summary"
    assert paper.ai_keywords == ["agents"]


def test_paper_info_preserves_zero_stars_from_nested_metadata() -> None:
    paper = PaperInfo(paper={"id": "2611.16961", "githubStars": 0})

    assert paper.github_stars == 0
