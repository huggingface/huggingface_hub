This document covers all steps that need to be done in order to do a release of the `huggingface_hub` library.

1. On a clone of the main repo, not your fork, checkout the main branch and pull the latest changes:
```
git checkout main
git pull
   ```

2. Checkout a new branch with the version that you'd like to release: v<MINOR-VERSION>-release,
for example `v0.5-release`. All patches will be done to that same branch.

3. Update the `__version__` variable in the `src/huggingface_hub/__init__.py` file to point
to the version you're releasing:
```
__version__ = "<VERSION>"
   ```

4. Make sure that the conda build works correctly by building it locally:
```
conda install -c defaults anaconda-client conda-build
HUB_VERSION=<VERSION> conda-build .github/conda
   ```

5. Make sure that the pip wheel works correctly by building it locally and installing it:
```
pip install setuptools wheel
python setup.py sdist bdist_wheel
pip install dist/huggingface_hub-<VERSION>-py3-none-any.whl
   ```

6. Commit, tag, and push the branch:
```
git commit -am "Release: v<VERSION>"
git tag v<VERSION> -m "Adds tag v<VERSION> for pypi and conda"
git push -u --tags origin v<MINOR-VERSION>-release
   ```

7. Verify that the docs have been built correctly. You can check that on the following link:
https://huggingface.co/docs/huggingface_hub/v<VERSION>

8. Checkout main once again to update the version in the `__init__.py` file:
```
git checkout main
   ```

9. Update the version to contain the `.dev0` suffix:
```
__version__ = "<VERSION+1>.dev0"  # For example, after releasing v0.5.0 or v0.5.1: "0.6.0.dev0".
   ```

10. Push the changes!
```
git push origin main
```
