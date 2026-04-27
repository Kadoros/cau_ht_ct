import hashlib
import os


def get_official_hash(folder_path):
    file_hashes = []
    for root, dirs, files in os.walk(folder_path):
        for file in sorted(files):
            if file.startswith("."):
                continue
            file_path = os.path.join(root, file)
            hasher = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            file_hashes.append(hasher.hexdigest().upper())
    combined = "".join(file_hashes)
    final_hash = hashlib.md5(combined.encode()).hexdigest().upper()
    return final_hash


print("train:", get_official_hash("./train"))
print("model:", get_official_hash("./best_model.pth"))
