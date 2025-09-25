<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->


# OAuth and FastAPI

OAuth is an open standard for access delegation, commonly used to grant applications limited access to a user's information without exposing their credentials. When combined with FastAPI it allows you to build secure APIs that allow users to log in using external identity providers like Google or GitHub.
In a usual scenario:
- FastAPI will define the API endpoints and handles the HTTP requests.
- OAuth is integrated using libraries like fastapi.security or external tools like Authlib.
- When a user wants to log in, FastAPI redirects them to the OAuth provider’s login page.
- After successful login, the provider redirects back with a token.
- FastAPI verifies this token and uses it to authorize the user or fetch user profile data.

This approach helps avoid handling passwords directly and offloads identity management to trusted providers.

# Hugging Face OAuth Integration in FastAPI

This module provides tools to integrate Hugging Face OAuth into a FastAPI application. It enables user authentication using the Hugging Face platform including mocked behavior for local development and real OAuth flow for Spaces.



## OAuth Overview

The `attach_huggingface_oauth` function adds login, logout, and callback endpoints to your FastAPI app. When used in a Space, it connects to the Hugging Face OAuth system. When used locally it will inject a mocked user. Click here to learn more about [adding a Sign-In with HF option to your Space](https://huggingface.co/docs/hub/en/spaces-oauth)


### How to use it?

```python
from huggingface_hub import attach_huggingface_oauth, parse_huggingface_oauth
from fastapi import FastAPI, Request

app = FastAPI()
attach_huggingface_oauth(app)

@app.get("/")
def greet_json(request: Request):
    oauth_info = parse_huggingface_oauth(request)
    if oauth_info is None:
        return {"msg": "Not logged in!"}
    return {"msg": f"Hello, {oauth_info.user_info.preferred_username}!"}
```

> [!TIP]
> You might also be interested in [a practical example that demonstrates OAuth in action](https://huggingface.co/spaces/Wauplin/fastapi-oauth/blob/main/app.py).
> For a more comprehensive implementation, check out [medoidai/GiveBackGPT](https://huggingface.co/spaces/medoidai/GiveBackGPT) Space which implements HF OAuth in a full-scale application.


### attach_huggingface_oauth

[[autodoc]] attach_huggingface_oauth

### parse_huggingface_oauth

[[autodoc]] parse_huggingface_oauth

### OAuthOrgInfo

[[autodoc]] OAuthOrgInfo

### OAuthUserInfo

[[autodoc]] OAuthUserInfo

### OAuthInfo

[[autodoc]] OAuthInfo
