from hf_api import HfApi, HfFolder

token = HfFolder.get_token()
api = HfApi().update_repo_visibility(token, "adapter", private=True)
