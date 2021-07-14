# Clone the repository
git-lfs install
git clone "https://huggingface.co/${MODEL_ID}"

# Get repo name: osanseviero/my-model -> my-model and install
# requirements
REPO_NAME=${MODEL_ID#*/}  
REQUIREMENTS="${REPO_NAME}/requirements.txt"
if [ -f "${REQUIREMENTS}" ]; then
    pip install --no-cache-dir -r "${REQUIREMENTS}"
fi

# Move files so they are available in the pipelines.
mv ${REPO_NAME}/* .

python app/main.py
