import hashlib
import os


def get_official_hash(path):
    file_hashes = []

    # 1. 입력 경로가 폴더인 경우 (데이터셋용)
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
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

    # 2. 입력 경로가 단일 파일인 경우 (모델용)
    elif os.path.isfile(path):
        hasher = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        final_hash = hasher.hexdigest().upper()

    else:
        return "ERROR: Path not found"

    return final_hash


# 실행
print("train 폴더 해시:", get_official_hash("./train"))
print(
    "Aggressive 모델 해시:",
    get_official_hash("/Users/kadoros1130/WS/1_CAU/HKT/CT/sub_v4_normal/model.pth"),
)
print(
    "Stable 모델 해시:",
    get_official_hash("/Users/kadoros1130/WS/1_CAU/HKT/CT/sub_v5_safe/model.pth"),
)
