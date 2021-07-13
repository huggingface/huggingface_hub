git clone "https://huggingface.co/${MODEL_ID}"

# Get repo name: osanseviero/my-model -> my-model, clone requirements
# and move to corresponding directory.
REPO_NAME=${MODEL_ID#*/}  
REQUIREMENTS="${REPO_NAME}/requirements.txt"
if [ -f "${REQUIREMENTS}" ]; then
    pip install --no-cache-dir -r "${REQUIREMENTS}"
fi
rm -rf src/
mv ${REPO_NAME}/* .
python app/main.py
