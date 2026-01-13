import random
import string
import tempfile
from pathlib import Path

from huggingface_hub import BucketAddFile, BucketDeleteFile, HfApi


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
with tempfile.TemporaryDirectory() as temp_dir:
    add_operations = []
    for i in range(10):
        file_name = "".join(random.choices(string.ascii_letters + string.digits, k=10)) + ".bin"
        file_path = Path(temp_dir) / file_name
        file_path.write_text(
            f"This is a test file {i}" + "".join(random.choices(string.ascii_letters + string.digits, k=100)) * 1000
        )
        add_operations.append(
            BucketAddFile(path_in_repo=file_path.relative_to(temp_dir).as_posix(), path_or_fileobj=file_path)
        )
    print(api.batch_bucket_files(bucket_id=bucket_id, operations=add_operations))

print("\n# List bucket tree (with files)")
objects = list(api.list_bucket_tree(bucket_id=bucket_id))
print(f"Found {len(objects)} objects in bucket: {[obj['path'] for obj in objects]}")

bucket_file = objects[0]["path"]

print(f"\n# Head bucket /resolve {bucket_file}")
print(dict(api.head_bucket_file(bucket_id=bucket_id, remote_path=bucket_file).headers))


print("\n# Delete first 3 files")
delete_operations = [BucketDeleteFile(path_in_repo=obj["path"]) for obj in objects[:3]]
print(api.batch_bucket_files(bucket_id=bucket_id, operations=delete_operations))

print(f"\n# Head bucket /resolve {bucket_file} (doesn't exist anymore)")
try:
    api.head_bucket_file(bucket_id=bucket_id, remote_path=bucket_file).headers
except Exception as e:
    print(e)

print("\n# List bucket tree (with files)")
objects = list(api.list_bucket_tree(bucket_id=bucket_id))
print(f"Found {len(objects)} objects in bucket: {[obj['path'] for obj in objects]}")
bucket_file = objects[0]["path"]

print(f"\n# Download file {bucket_file} from bucket")
api.download_bucket_file(bucket_id=bucket_id, remote_path=bucket_file, local_path=bucket_file)
print(f"File downloaded to {bucket_file}")
print(f"Local file size: {Path(bucket_file).stat().st_size} bytes")
Path(bucket_file).unlink()
api.delete_bucket(bucket_id="julien-c/test-bucket-with-files", missing_ok=True)
