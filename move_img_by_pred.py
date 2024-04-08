import os
import json
from shutil import move

# 결과 JSON 파일 로드
with open('classification_results.json', 'r') as f:
    results = json.load(f)

# 기본 이미지 폴더와 대상 폴더
base_folder = 'path_to_your_images'
target_base_folder = 'classified'

# 각 결과에 따라 이미지 이동
for result in results:
    src_path = os.path.join(base_folder, result['filename'])
    target_folder = os.path.join(target_base_folder, result['folder'])
    target_path = os.path.join(target_folder, result['filename'])

    # 대상 폴더가 없으면 생성
    os.makedirs(target_folder, exist_ok=True)

    # 파일 이동
    move(src_path, target_path)

print("Files moved based on classification.")
