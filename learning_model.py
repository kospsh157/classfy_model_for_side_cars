# pip install tensorflow
# pip install numpy
# pip install matplotlib


import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v4 import preprocess_input

# JSON 파일에서 이미지 경로와 라벨 읽기
def load_data(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

data = load_data('labeled_images.json')

# 데이터를 학습용과 검증용으로 분리
from sklearn.model_selection import train_test_split
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# ImageDataGenerator 생성
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

def create_generator(data, datagen):
    while True:
        for item in data:
            image = tf.keras.preprocessing.image.load_img(item['filename'], target_size=(299, 299))
            image = tf.keras.preprocessing.image.img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = datagen.preprocess_input(image)
            label = item['label']
            yield (image, label)

# 생성기 설정
train_generator = create_generator(train_data, train_datagen)
val_generator = create_generator(val_data, val_datagen)





# 데이터 구성 및 컴파일
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionV4

# Inception-V4 모델 로드
base_model = InceptionV4(weights='imagenet', include_top=False)

# 상위 레이어 추가
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)  # 3개의 출력 라벨

model = Model(inputs=base_model.input, outputs=predictions)

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])






# 모델 학습
# 학습 파라미터 설정
epochs = 10
batch_size = 32

# 모델 학습
model.fit(train_generator, steps_per_epoch=len(train_data)//batch_size, epochs=epochs,
          validation_data=val_generator, validation_steps=len(val_data)//batch_size)
# 모델 평가
model.evaluate(val_generator, steps=len(val_data))


# 모델 저장
model.save('saved_models/trained_model.h5')
print("Model saved as trained_model.h5")




# 1. 100장씩 인간이 설별하여 3개의 폴더에 저장한다. 
# 2. 모델로 3개의 폴더와 라벨링json파일과 같이 학습 한다. 
# 3. 평가해보고, 지표 알아본다음에, 인간이 살펴본다.
# 4. 제대로 모델이 분류를 한다면,
# 5. 다음과 같이 분류를 진행한다.
    # 1. side of cars로 대량으로 크롤링 한다. (SUV, sedan, truck)
    # 2. 바퀴 또한 대량으로 크롤링 한다.(이 부분에 대해서는 어떻게 활용하는 것인지 희정 연구원님께 다시 물어봐야함.)
    # 3. 학습된 모델을 가지고 299*299로 바꿔서 분류를 진행한다.
    # 4. 분류 결과는 라벨링json파일과 같이 라벨링으로 출력한다.
    # 5. 파일명:라벨링 된 json출력물을 이용해서 이미지별로 분류 라벨링에 맞는 폴더로 이동시킨다. 
    # 6. 분류 된 폴더로 가서 맞게 분류 했는지 인간이 살펴본다. 

