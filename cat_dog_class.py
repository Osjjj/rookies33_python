import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import numpy as np

# 모델 로드
def load_model() :
    try :        
        return tf.keras.models.load_model('cat_dog_classifier.keras')
        st.success('모델을 로드했습니다.', icon='🚨')
    except:
        st.error('모델을 로드할수 없습니다. 경로를 확인해주세요.', icon="🚨")
        #raise e
    
model = load_model()

# 업로된 이미지의 전처리
def preprocess_image(image):
    try:
        image = image.resize( (150, 150) ) # 이미지 크기 조정
        image = np.array(image) / 255.0  # 정규화
        if image.shape[-1] != 3:
            raise ValueError('이미지는 RGB 형식의 컬러 이미지여야 합니다.')
        image = np.expand_dims(image, axis=0)  # (1, 150, 150, 3)
        return  image
    except Exception as e :
        st.error(f'이미지 전처리중 오류 발생 : {e}', icon="🚨")
        return None

# UI
st.title('개/고양이 분류기')
st.write('이미지를 업로드 하면 개 또는 고양이를 판별합니다.')

upload_file = st.file_uploader('이미지를 업로드하세요!', type=['jpg', 'png', 'jpeg'])

if upload_file is not None :
    try :
        # 이미지 로드
        # 업로드 된 파일을 이미지 객체로 변환
        image = Image.open(upload_file) 
        st.image(image, caption='업로드 한 이미지', use_container_width=True)

        preprocessed_image = preprocess_image(image) # 이미지 전처리
        
        if preprocessed_image is not None :
            prediction = model.predict(preprocessed_image)
            print(prediction)

            # 결과 표시
            if prediction[0][0] > 0.5 :
                st.success('이 이미지는 개로 분류되었습니다.')
            else :
                st.success('이 이미지는 고양이로 분류되었습니다.')
    except UnidentifiedImageError :  # 이미지 형식이 아닌경우 발생
        st.error('이미지를 로드할 수 없습니다. 지원되지 않는 파일 형식입니다.', icon="🚨")
    except Exception as e :
        st.error(f'예측 처리중 오류 발생 : {e}', icon="🚨")

st.info('note : 이 모델은 학습된 데이터에 따라 결과가 달라질수 있습니다.')