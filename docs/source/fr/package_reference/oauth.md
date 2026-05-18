<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->


# OAuth et FastAPI

OAuth est un standard ouvert pour la délégation d'accès, couramment utilisé pour accorder aux applications un accès limité aux informations d'un utilisateur sans exposer ses identifiants. Combiné avec FastAPI, il vous permet de construire des APIs sécurisées qui permettent aux utilisateurs de se connecter en utilisant des fournisseurs d'identité externes comme Google ou GitHub.
Dans un scénario habituel :
- FastAPI définira les endpoints API et gérera les requêtes HTTP.
- OAuth est intégré en utilisant des bibliothèques comme fastapi.security ou des outils externes comme Authlib.
- Lorsqu'un utilisateur souhaite se connecter, FastAPI le redirige vers la page de connexion du fournisseur OAuth.
- Après une connexion réussie, le fournisseur redirige en retour avec un token.
- FastAPI vérifie ce token et l'utilise pour autoriser l'utilisateur ou récupérer les données de profil utilisateur.

Cette approche aide à éviter de gérer les mots de passe directement et délègue la gestion d'identité à des fournisseurs de confiance.

# Intégration Hugging Face OAuth dans FastAPI

Ce module fournit des outils pour intégrer Hugging Face OAuth dans une application FastAPI. Il permet l'authentification utilisateur en utilisant la plateforme Hugging Face, incluant un comportement mocké pour le développement local et un flux OAuth réel pour les Spaces.



## Aperçu OAuth

La fonction `attach_huggingface_oauth` ajoute des endpoints de connexion, déconnexion et callback à votre app FastAPI. Lorsqu'elle est utilisée dans un Space, elle se connecte au système OAuth de Hugging Face. Lorsqu'elle est utilisée localement, elle injectera un utilisateur mocké. Cliquez ici pour en savoir plus sur [l'ajout d'une option Sign-In with HF à votre Space](https://huggingface.co/docs/hub/en/spaces-oauth)


### Comment l'utiliser ?

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
> Vous pourriez également être intéressé par [un exemple pratique qui démontre OAuth en action](https://huggingface.co/spaces/Wauplin/fastapi-oauth/blob/main/app.py).
> Pour une implémentation plus complète, consultez le Space [medoidai/GiveBackGPT](https://huggingface.co/spaces/medoidai/GiveBackGPT) qui implémente HF OAuth dans une application à grande échelle.


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
