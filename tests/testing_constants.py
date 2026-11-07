USER = "__DUMMY_HUB_USER__"
FULL_NAME = "Dummy User"
PASS = "__DUMMY_TRANSFORMERS_PASS__"

# Not critical, only usable on the sandboxed CI instance.
TOKEN = "hf_HubCITokenXXXXXXXXXXXXXXXXXXXXX"

# Used to create repos that we don't own (example: for gated repo)
# Token is not critical. Also public in https://github.com/huggingface/datasets-server
OTHER_USER = "DVUser"
OTHER_TOKEN = "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"

# Used to test enterprise features, typically creating private repos by default
ENTERPRISE_USER = "EnterpriseAdmin"
ENTERPRISE_ORG = "EnterpriseOrgPrivate"
ENTERPRISE_TOKEN = "hf_enterprise_admin_token"

ENDPOINT_PRODUCTION = "https://huggingface.co"
ENDPOINT_STAGING = "https://hub-ci.huggingface.co"

ENDPOINT_PRODUCTION_URL_SCHEME = ENDPOINT_PRODUCTION + "/{repo_id}/resolve/{revision}/{filename}"

# Example model ids

# An actual model hosted on huggingface.co,
# w/ more details.
DUMMY_MODEL_ID = "julien-c/dummy-unknown"
DUMMY_MODEL_ID_REVISION_ONE_SPECIFIC_COMMIT = "f2c752cfc5c0ab6f4bdec59acea69eefbee381c2"
# One particular commit (not the top of `main`)

# "hf-internal-testing/dummy-will-be-renamed" has been renamed to "hf-internal-testing/dummy-renamed"
DUMMY_RENAMED_OLD_MODEL_ID = "hf-internal-testing/dummy-will-be-renamed"

SAMPLE_DATASET_IDENTIFIER = "lhoestq/custom_squad"
# Example dataset ids
DUMMY_DATASET_ID = "gaia-benchmark/GAIA"
DUMMY_DATASET_ID_REVISION_ONE_SPECIFIC_COMMIT = "c603981e170e9e333934a39781d2ae3a2677e81f"  # on branch "test-branch"

# Xet testing
DUMMY_XET_MODEL_ID = "celinah/dummy-xet-testing"
DUMMY_XET_FILE = "dummy.safetensors"

# extra large file for testing on production
DUMMY_EXTRA_LARGE_FILE_MODEL_ID = "brianronan/dummy-xet-edge-case-files"
DUMMY_EXTRA_LARGE_FILE_NAME = "verylargemodel.safetensors"  # > 50GB file
