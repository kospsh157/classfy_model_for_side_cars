import os

# 이미지가 저장된 폴더 지정
directory = './side_cars_view'

# 폴더 내의 모든 파일을 순회하며 이름 변경
for index, filename in enumerate(os.listdir(directory)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 이미지 파일 형식 필터링
        # 새 파일명 생성
        new_filename = f"car_side_{index + 1}.jpg"
        # 원래 파일의 전체 경로
        old_file = os.path.join(directory, filename)
        # 새 파일의 전체 경로
        new_file = os.path.join(directory, new_filename)
        # 파일 이름 변경
        os.rename(old_file, new_file)
        print(f"Renamed '{filename}' to '{new_filename}'")
