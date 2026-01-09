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
print("TODO: doesn't work yet, see comments in the code")
# TODO: download not working
#       server-side issue: {"message":"error getting file reconstruction, error: InvalidRange(\"range start is greater than file length\")","caller":"cas_server/src/handler.rs:197"}
#       {"message":"Request failed","total_latency_ms":2,"session_id":"01KEHE6F97PS81FWV3XZ06MGPZ","remote_ip":"","http.method":"GET","http.target":"/reconstructions/2d32e4348fc3d156bc25b52dec286da152b773884115988245f2f974b5567147","http.route":"/reconstructions/{file_id}","http.status":416,"http.range":"1073741824-2147483648","repo":"6960e6943d6ac2452045df70","user":"5dd96eb166059660ed1ee413","reason":"status code: 416 Range Not Satisfiable","completion_reason":"ServerError"}
#       client-side issue:
#       RuntimeError: Data processing error: CAS service error : ReqwestMiddleware Error: Request failed after 5 retries
#
# When calling http://localhost:4884/v1/reconstructions/2d32e4348fc3d156bc25b52dec286da152b773884115988245f2f974b5567147 manually I'm getting:
# {
#   "offset_into_first_range": 0,
#   "terms": [
#     {
#       "hash": "ac6ad16275b9f2a036f8f10e51671c180c2dada906ba9691ab6c69a9f83631ae",
#       "unpacked_length": 950982,
#       "range": {
#         "start": 613,
#         "end": 627
#       }
#     }
#   ],
#   "fetch_info": {
#     "ac6ad16275b9f2a036f8f10e51671c180c2dada906ba9691ab6c69a9f83631ae": [
#       {
#         "range": {
#           "start": 613,
#           "end": 627
#         },
#         "url": "http://host.docker.internal:9000/cas/xorbs/default/ac6ad16275b9f2a036f8f10e51671c180c2dada906ba9691ab6c69a9f83631ae?x-id=GetObject&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minio%2F20260109%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260109T120133Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host%3Brange&X-Amz-Signature=fbf632af62298b282edcd3baf31ef4a734ced25e6245707c1f9e1b4e05ce446d",
#         "url_range": {
#           "start": 37770374,
#           "end": 38719071
#         }
#       }
#     ]
#   }
# }
# api.download_bucket_file(bucket_id=bucket_id, remote_path=bucket_file, local_path=bucket_file)
# print(f"File downloaded to {bucket_file}")
# print(f"Local file size: {Path(bucket_file).stat().st_size} bytes")
# Path(bucket_file).unlink()
api.delete_bucket(bucket_id="julien-c/test-bucket-with-files", missing_ok=True)
