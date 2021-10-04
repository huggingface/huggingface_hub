---
title: Security and trust on the Hugging Face Hub
---

## Signing your commits with GPG

`git` has an authentication layer to control who can push commits to a repo, but it does not authentify the actual commit authors.

In other words, you can commit changes as `Elon Musk <elon@tesla.com>`, push them to your preferred Git host (for instance github.com) and your commit will link to Elon's GitHub profile. (Try it! But don't blame us if Elon gets mad at you for impersonating him)

See this post by Ale Segala for more context: [How (and why) to sign Git commits](https://withblue.ink/2020/05/17/how-and-why-to-sign-git-commits.html)

Using GPG however – and a key server – you can "prove" that a commit was authored by *you*. We'll explain how to set this up on hf.co below.

The Pro Git book is, as usual, a good resource about commit signing: [Pro Git: Signing your work](https://git-scm.com/book/en/v2/Git-Tools-Signing-Your-Work).


### Why we implemented signed commits verification on the Hugging Face hub

The reasons we implemented GPG signing was:
- to provide finer-grained security, especially as more and more Enterprise users rely on the Hub (in the future you'll be able to disallow non-signed commits)
- coupled with our model [eval results metadata](https://github.com/huggingface/huggingface_hub/blame/main/modelcard.md), this enables a cryptographically-secure trustable source of ML benchmarks 🤯 

### Setting up signed commits verification

> You will need to install [GPG](https://gnupg.org/) on your system in order to execute the following commands.
> It's included by default in most Linux distribution.
> On Windows, it is included in Git Bash that comes with git for Windowss.

You can sign your commits locally using [GPG](https://gnupg.org/).
You can then configure your profile to mark these commits as **verified** on the HuggingFace hub,
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

If you have no GPG keypair, or you don't want to use the existing keys to sign your commits, go to **Generating a new GPG key**.

Otherwise, go straight to **Adding a GPG key to your account**.

### Generating a new GPG key

To generate a GPG key, run the following:

```bash
gpg --gen-key
```

GPG will then guide you through the process of creating a GPG key pair.

Make sure you specify an email address for this key, and that the email address matches the one you specified in your Hugging Face [account](https://huggingface.co/settings/account).

### Adding a GPG key to your account

First, select or generate a GPG key on your computer. Make sure that the e-mail address of the key matches the one of your Hugging Face [account](https://huggingface.co/settings/account),
and that the e-mail of your account is verified.

Then, export the public part of the selected key:

```bash
gpg --armor --export <YOUR KEY ID>
```

Then visit your profile [settings page](https://huggingface.co/settings/keys) and click on **Add GPG Key**.

Copy & paste the output of the `gpg --export` command in the text area and click on **Add Key**.

Congratulations, you've just added a GPG key to your account !

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

Once pushed on the huggingface hub, you should see the commit with a "Verified" badge.
