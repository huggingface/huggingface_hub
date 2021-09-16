USER = "__DUMMY_TRANSFORMERS_USER__"
FULL_NAME = "Dummy User"
PASS = "__DUMMY_TRANSFORMERS_PASS__"

ENDPOINT_PRODUCTION = "https://huggingface.co"
ENDPOINT_STAGING = "https://moon-staging.huggingface.co"
ENDPOINT_STAGING_BASIC_AUTH = f"https://{USER}:{PASS}@moon-staging.huggingface.co"

ENDPOINT_PRODUCTION_URL_SCHEME = ENDPOINT_PRODUCTION + "/{repo_id}/resolve/{revision}/{filename}"
