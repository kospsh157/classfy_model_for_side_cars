import os
import json

# 각 폴더와 해당 폴더의 라벨 매핑
folders = {
    'rear_cars': 'rear',
    'front_cars': 'front',
    'wheels': 'wheel',
    'side_cars': 'side'
}

# 최종 데이터를 저장할 리스트
data = []

# 각 폴더를 순회하면서 이미지 파일과 라벨을 매핑
for folder, label in folders.items():
    # 폴더 내 모든 파일 확인
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 이미지 파일 필터링
            # 파일의 전체 경로 생성
            file_path = os.path.join(folder, filename)
            # 데이터 리스트에 파일 정보와 라벨 추가
            data.append({'filename': file_path, 'label': label})

# JSON 파일로 결과 저장
with open('labeled_images.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)

print("Labeling completed and saved to labeled_images.json")
