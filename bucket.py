from pathlib import Path
from pprint import pprint

from huggingface_hub import HfApi


token = "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"  # local write token
endpoint = "http://localhost:5173"
username = "julien-c"

api = HfApi(endpoint=endpoint, token=token)

print("# Create bucket")
output = api.create_bucket(bucket_id="test-bucket", exist_ok=True)
print(output)
bucket_id = output["id"]

print("\n# Get bucket info")
print(api.bucket_info(bucket_id=bucket_id))

print("\n# List buckets (non-empty)")
print(api.list_buckets())

print("\n# Delete bucket")
print(api.delete_bucket(bucket_id=bucket_id))

print("\n# List buckets (empty)")
print(api.list_buckets())

print("\n# Create private bucket")
output = api.create_bucket(bucket_id="test-bucket-private", private=True, exist_ok=True)
print(output)
bucket_id = output["id"]

print("\n# Get private bucket info (with token)")
print(api.bucket_info(bucket_id=bucket_id))

print("\n# Get private bucket info (without token)")
try:
    api.bucket_info(bucket_id=bucket_id, token=False)
except Exception as e:
    print(e)
api.delete_bucket(bucket_id=bucket_id)

print("\n# Create bucket again")
api.delete_bucket(bucket_id="julien-c/test-bucket-with-files", missing_ok=True)
output = api.create_bucket(bucket_id="test-bucket-with-files")
print(output)
bucket_id = output["id"]

print("\n# List bucket tree (empty)")
print(list(api.list_bucket_tree(bucket_id=bucket_id)))

print("\n# Upload file to bucket")
local_dir = Path("/home/wauplin/projects/huggingface_hub/downloads")
files = [
    (
        str(path),  # local_path
        path.relative_to(local_dir).as_posix(),  # remote_path
    )
    for path in local_dir.glob("*")
][:5]  # limit to 5 files for the example
print(api.upload_bucket_files(bucket_id=bucket_id, files=files))

print("\n# List bucket tree (with files)")
objects = list(api.list_bucket_tree(bucket_id=bucket_id))
pprint(objects)

bucket_file = objects[0]["path"]
print(f"\n# Head bucket /resolve {bucket_file}")
print(dict(api.head_bucket_file(bucket_id=bucket_id, remote_path=bucket_file).headers))

print(f"\n# Download file {bucket_file} from bucket")
print(
    "TODO: doesn't work in local... (downloading a newly uploaded Xet file doesn't work, both from a model or bucket)"
)
# api.download_bucket_file(bucket_id=bucket_id, remote_path=bucket_file, local_path=bucket_file)
# print(f"File downloaded to {bucket_file}")
# print(f"Local file size: {Path(bucket_file).stat().st_size} bytes")
# Path(bucket_file).unlink()
api.delete_bucket(bucket_id="julien-c/test-bucket-with-files", missing_ok=True)
