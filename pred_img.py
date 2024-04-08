import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v4 import preprocess_input

# 모델 로드
model = tf.keras.models.load_model('path_to_your_model.h5')

# 이미지 폴더
image_folder = 'path_to_your_images'
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

# 분류 결과를 저장할 리스트
classification_results = []

for img_filename in image_files:
    img_path = os.path.join(image_folder, img_filename)
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # 예측 수행
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=-1)[0]

    # 결과 매핑
    classes = ['rear', 'front', 'side', 'wheel']
    predicted_label = classes[predicted_class]

    # 결과 저장
    classification_results.append({
        "filename": img_filename,
        "folder": predicted_label
    })

# JSON 파일로 저장
with open('classification_results.json', 'w') as f:
    json.dump(classification_results, f, indent=4)

print("Classification complete and saved to classification_results.json")
