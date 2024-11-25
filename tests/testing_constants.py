USER = "__DUMMY_TRANSFORMERS_USER__"
FULL_NAME = "Dummy User"
PASS = "__DUMMY_TRANSFORMERS_PASS__"

# Not critical, only usable on the sandboxed CI instance.
TOKEN = "hf_94wBhPGp6KrrTH3KDchhKpRxZwd6dmHWLL"

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
