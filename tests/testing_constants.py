import os


USER = "__DUMMY_TRANSFORMERS_USER__"
FULL_NAME = "Dummy User"
PASS = "__DUMMY_TRANSFORMERS_PASS__"

# Not critical, only usable on the sandboxed CI instance.
TOKEN = "hf_94wBhPGp6KrrTH3KDchhKpRxZwd6dmHWLL"

# Used to create repos that we don't own (example: for gated repo)
# Token is not critical. Also public in https://github.com/huggingface/datasets-server
OTHER_USER = "__DUMMY_DATASETS_SERVER_USER__"
OTHER_TOKEN = "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"

ENDPOINT_PRODUCTION = "https://huggingface.co"
ENDPOINT_STAGING = "https://hub-ci.huggingface.co"

ENDPOINT_PRODUCTION_URL_SCHEME = ENDPOINT_PRODUCTION + "/{repo_id}/resolve/{revision}/{filename}"

# Token to be set as environment variable.
# Almost all features are tested on staging environment. However, Spaces are not supported
# there which makes it impossible to test secrets/hardware requests.
#
# Value is set as a secret in Github actions. To make this test work locally, set
# `HUGGINGFACE_HUB_PRODUCTION_TEST` environment variable on your machine with a personal
# token (see https://huggingface.co/settings/tokens). The test pipeline will only create
# private spaces (and delete them afterwards).
#
# See `fx_production_space` fixture. Goal is to limit its usage as much as possible.
PRODUCTION_TOKEN = os.environ.get("HUGGINGFACE_PRODUCTION_USER_TOKEN")
