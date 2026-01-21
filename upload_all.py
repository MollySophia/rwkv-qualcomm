import os
OUTPUT_DIR = "output"
VERSION = "2.42-260118"
FILTERS = []

def upload_all():
    for file in os.listdir(OUTPUT_DIR):
        if file.endswith(".rmpack"):
            upload_file(file)

uploaded = []

def upload_file(file):
    for filter in FILTERS:
        if filter not in file:
            return

    remote_path = f"/qnn/{VERSION}/{file}"
    if "modrwkv" in file:
        remote_path = f"/multimodal/model/modrwkv-v3/{VERSION}/{file}"
    elif "respark" in file:
        remote_path = f"/multimodal/sparktts/{VERSION}/{file}"
    elif "ABC" in file:
        return

    print(f"Uploading {file}")
    result = os.system(f"huggingface-cli upload mollysama/rwkv-mobile-models output/{file} {remote_path}")
    if result != 0:
        print(f"Error uploading {file}: {result}")
        exit(result)
    
    uploaded.append(file)

upload_all()

total_size = 0
for file in uploaded:
    total_size += os.path.getsize(f"{OUTPUT_DIR}/{file}")
print(f"Total uploaded size: {total_size / 1024 / 1024} MB")
