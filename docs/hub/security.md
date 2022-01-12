---
title: Security and trust on the Hugging Face Hub
---

# Security and trust on the Hugging Face Hub

## User Access Tokens

### What are User Access Tokens?

User Access Tokens are the preferred way to authenticate an application or notebook to Hugging Face services. You can manage your access tokens in your [settings](https://huggingface.co/settings/token).

![/docs/assets/hub/access-tokens.png](/docs/assets/hub/access-tokens.png)

Access tokens allow applications and notebooks to perform specific actions specified by the scope of the roles shown in the following:

- `read`: tokens with this role can only be used to provide read access to repositories you could read. That includes public repositories and private repositories that you, or an organization you're a member of, own. Use this role if you only need to read content from the Hugging Face Hub (e.g. when downloading private models or doing inference).

- `write`: tokens with this role additionally grant write access to the repositories you have write access to. Use this token if you need to create or push content to a repository (e.g., when training a model or modifying a model card).

### How to manage User Access Tokens?

To create an access token, go to your settings, then click on the ["Access Tokens" tab](https://huggingface.co/settings/token). Click on the "New token" button to create a new User Access Token.

![/docs/assets/hub/new-token.png](/docs/assets/hub/new-token.png)

Select a role and a name for your token and voilà - you're ready to go!

You can delete and refresh User Access Tokens by clicking on the "Manage" button.

![/docs/assets/hub/delete-token.png](/docs/assets/hub/delete-token.png)

### How to use User Access Tokens?

There are plenty of ways to use a User Access Token to access the Hugging Face Hub, granting you the flexibility you need to build awesome apps on top of it.

- User Access Tokens can be used **in place of a password** to access the Hugging Face Hub with git or with basic authentication
- User Access Tokens can be passed as a **bearer token** when calling the [Inference API](https://huggingface.co/inference-api).
- User Access Tokens can be used in the Hugging Face Python libraries, such as `transformers` or `datasets`:

```python
from transformers import AutoModel

access_token = "hf_..."

model = AutoModel.from_pretrained("private/model", use_auth_token=access_token)
```

⚠️ Try not to leak your token! Though you can always rotate it, anyone will be able to read or write your private repos in the meantime which is 💩

### Best practices

We recommend you create one access token per app or usage. For instance, you could have a separate token for:
 * A local machine.
 * A Colab notebook.
 * An awesome custom inference server. 
 
 This way, you can invalidate one token without impacting your other usages.

We also recommend only giving the appropriate role to each token you create. If you only need read access (i.e., loading a dataset with the `datasets` library or retrieving the weights of a model), only give your access token the `read` role.

## Access Control in Organizations

Members of organizations can have three different roles: `read`, `write` or `admin`.

- `read` **role**:read-only access to the Organization's repos and metadata / settings (eg, the Organizations' profile, members list, API token, etc).

- `write` **role**: additional write rights to the Organization's repos. They can create, delete or rename any repo in the Organization namespace. They can also edit and delete files with the browser editor and push content with `git`.

- `admin` **role**: In addition to write rights on repos, admin members can update the Organization's profile, refresh the Organization's API token, and manage the Organization members.

As an organization `admin`, go to the "Members" section of the org settings to manage roles for users.

![/docs/assets/hub/org-members-page.png](/docs/assets/hub/org-members-page.png)

## Signing your commits with GPG

`git` has an authentication layer to control who can push commits to a repo, but it does not authentify the actual commit authors.

In other words, you can commit changes as `Elon Musk <elon@tesla.com>`, push them to your preferred `git` host (for instance github.com) and your commit will link to Elon's GitHub profile. (Try it! But don't blame us if Elon gets mad at you for impersonating him)

See this post by Ale Segala for more context: [How (and why) to sign `git` commits](https://withblue.ink/2020/05/17/how-and-why-to-sign-git-commits.html)

You can prove a commit was authored by you, using GNU Privacy Guard (GPG) and a key server. GPG is a cryptographic tool used to verify the authenticity of a message's origin. We'll explain how to set this up on hf.co below.

The Pro Git book is, as usual, a good resource about commit signing: [Pro Git: Signing your work](https://git-scm.com/book/en/v2/Git-Tools-Signing-Your-Work).


### Why we implemented signed commits verification on the Hugging Face hub

The reasons we implemented GPG signing were:
- To provide finer-grained security, especially as more and more Enterprise users rely on the Hub (in the future you'll be able to disallow non-signed commits)
- Coupled with our model [eval results metadata](https://github.com/huggingface/huggingface_hub/blame/main/modelcard.md), GPG signing enables a cryptographically-secure trustable source of ML benchmarks. 🤯 

### Setting up signed commits verification

> You will need to install [GPG](https://gnupg.org/) on your system in order to execute the following commands.
> It's included by default in most Linux distributions.
> On Windows, it is included in Git Bash (which comes with `git` for Windows).

You can sign your commits locally using [GPG](https://gnupg.org/).
Then configure your profile to mark these commits as **verified** on the Hub,
so other people can be confident that they come from a trusted source.

For a more in-depth explanation of how git and GPG interact, please visit the the [git documentation on the subject](https://git-scm.com/book/en/v2/Git-Tools-Signing-Your-Work)

Commits can have the following signing statuses:

| Status            | Explanation                                                  |
| ----------------- | ------------------------------------------------------------ |
| Verified          | The commit is signed and the signature is verified           |
| Unverified        | The commit is signed but the signature could not be verified |
| No signing status | The commit is not signed                                     |

For a commit to be marked as **verified**, you need to upload the public key used to sign it on your Hugging Face account.

Use the `gpg --list-secret-keys` command to list the GPG keys for which you have both a public and private key.
A private key is required for signing commits or tags.

If you have no GPG key pair, or you don't want to use the existing keys to sign your commits, go to **Generating a new GPG key**.

Otherwise, go straight to  [Adding a GPG key to your account](#adding-a-gpg-key-to-your-account).

### Generating a new GPG key

To generate a GPG key, run the following:

```bash
gpg --gen-key
```

GPG will then guide you through the process of creating a GPG key pair.

Make sure you specify an email address for this key, and that the email address matches the one you specified in your Hugging Face [account](https://huggingface.co/settings/account).

### Adding a GPG key to your account

1. First, select or generate a GPG key on your computer. Make sure that the email address of the key matches the one of your Hugging Face [account](https://huggingface.co/settings/account) and that the email of your account is verified.

2. Then, export the public part of the selected key:

```bash
gpg --armor --export <YOUR KEY ID>
```

3. Then visit your profile [settings page](https://huggingface.co/settings/keys) and click on **Add GPG Key**.

Copy & paste the output of the `gpg --export` command in the text area and click on **Add Key**.

4. Congratulations ! 🎉 You've just added a GPG key to your account !

### Configure git to sign your commits with GPG

The last step is to configure git to sign your commits:

```bash
git config user.signingkey <Your GPG Key ID>
git config user.email <Your email on hf.co>
```

You can then add the `-S` flag to your `git commit` commands to sign your commits !

```bash
git commit -S -m "My first signed commit"
```

Once pushed on the Hub, you should see the commit with a "Verified" badge.

## Malware Scanning

We run every file of your repositories through a [malware scanner](https://www.clamav.net/).

Scanning is triggered at each commit or when you visit a repository page.

Here is an [example view](https://huggingface.co/mcpotato/42-eicar-street/tree/main) of an infected file :

![/docs/assets/hub/eicar-hub-tree-view.png](/docs/assets/hub/eicar-hub-tree-view.png)

![/docs/assets/hub/eicar-hub-file-view.png](/docs/assets/hub/eicar-hub-file-view.png)

_Note_: if your file has neither an ok or infected badge, it could mean that it is either currently being scanned / waiting to be scanned or that there was an error during the scan. It can take up to a few minutes to be scanned.

