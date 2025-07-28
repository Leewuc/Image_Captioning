# Image Captioning

이 저장소는 Flickr8k 데이터셋을 활용하여 이미지를 보고 자연어 설명을 생성하는 모델을 구현합니다. DenseNet201을 이용해 이미지를 특징으로 변환하고, LSTM 기반 언어 모델로 캡션을 생성합니다.

## 프로젝트 구조

- **Module.py** - 공통으로 사용하는 라이브러리와 설정을 모아둔 파일
- **Image.py** - 이미지와 캡션 데이터를 불러오고 시각화하는 도구
- **Caption_text.py** - 캡션 전처리 및 토크나이저 구축
- **Feature_Extraction.py** - DenseNet201을 사용하여 이미지 특징 추출
- **Model.py** - 이미지 특징과 텍스트를 결합한 캡션 생성 모델 정의 및 학습
- **Inference.py** - 학습된 모델로 새로운 이미지에 대한 설명을 생성

## 시작하기

1. [Flickr8k 데이터셋](https://www.kaggle.com/datasets/adityajn105/flickr8k) 을 다운로드해 `../input/flickr8k/` 위치에 배치합니다.
2. 필요한 파이썬 패키지를 설치합니다. (TensorFlow 2.x 권장)
3. `Model.py`를 실행해 학습을 시작합니다. 학습이 완료되면 `model.h5` 파일이 생성됩니다.
4. `Inference.py`를 실행하면 테스트 이미지에 대한 캡션을 확인할 수 있습니다.

## 예시 결과

학습된 모델은 입력 이미지에 대해 다음과 같은 설명을 생성합니다:

```
startseq a man riding a surfboard on a wave endseq
```

## 참고

- 코드 구조는 Jupyter 노트북에서 실험한 내용을 스크립트 형태로 정리한 것입니다.
- 학습 및 추론 속도는 하드웨어 사양에 따라 달라질 수 있습니다.

즐거운 실험 되세요!
