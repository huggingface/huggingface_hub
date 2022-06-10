USER = "__DUMMY_TRANSFORMERS_USER__"
FULL_NAME = "Dummy User"
PASS = "__DUMMY_TRANSFORMERS_PASS__"

# Not critical, only usable on the sandboxed CI instance.
TOKEN = "hf_94wBhPGp6KrrTH3KDchhKpRxZwd6dmHWLL"

ENDPOINT_PRODUCTION = "https://huggingface.co"
ENDPOINT_STAGING = "https://hub-ci.huggingface.co"
ENDPOINT_STAGING_BASIC_AUTH = f"https://{USER}:{PASS}@hub-ci.huggingface.co"

ENDPOINT_PRODUCTION_URL_SCHEME = (
    ENDPOINT_PRODUCTION + "/{repo_id}/resolve/{revision}/{filename}"
)
