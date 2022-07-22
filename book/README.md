# 파이썬을 이용한 머신러닝, 딥러닝 실전 개발 입문(개정판)
* 쿠지라 히코우즈쿠에 지음, 윤인성 옮김, 위키북스 출판
* [Github Link](https://github.com/wikibook/pyml-rev)
* 다음의 컴퓨터 환경에서 소스코드를 실행했습니다.
1)	- CPU		: Intel Pentium Gold G6405
	- OS		: Ubuntu Server 22.04 LTS
	- Python	: Version 3.8, venv
2)	- CPU		: Apple Silicon M1
	- OS		: Mac
	- Python	: Version 3.9, Conda Miniforge3
3)	- CPU		: Intel i5-10400F
	- GPU		: Nvidia GTX 1660ti
	- OS		: Ubuntu Server 22.04 LTS
	- Python	: Version 3.8, Docker(tensorflow/tensorflow:latest-gpu-jupyter)

## 1장, 크롤링과 스크레이핑
### urllib.request
* urllib.request.urlretrieve()
* urllib.request.urlopen()
* QueryString을 이용해 데이터 보내기

### BeautifulSoup
``` python
soup = BeautifulSoup(html, 'html.parser')

title = soup.find(id='title')
links = soup.find_all('a')
for a in links:
    href = a.attrs['href']
    text = a.string
```

* CSS Selector
``` python
soup.select('#ve-list < li.black')
```

* 정규조합식

* 페이지에 링크된 자료까지 모두 받기(재귀 이용)
    - [source code](./ch1/cr-getall.py)

## 2장, 고급 스크레이핑
### requests
* requests.session()을 이용한 사이트 로그인
* requests.get()
* requests.post()
* requests.put()
* requests.delete()

### Selenium
* [Selenium with Python](http://selenium-python.readthedocs.io/index.html)
* [SeleniumHQ Documentation](http://docs.seleniumhq.org/docs)
	- [Selenium Github](https://github.com/SeleniumHQ/seleniumhq.github.io)

### 정기적인 크롤링
* cron (macOS and Linux)
    - `crontab -e`
    - It can alert using email
* Task Scheduler (Windows)

## 3장, 데이터 소스의 서식과 가공
### 데이터 형식
- 텍스트 데이터와 바이너리 데이터

* xml
* json
* yaml
* csv/tsv

#### openpyxl
- 엑셀 파일을 읽고 쓰게 해주는 파이썬 라이브러리

#### pandas

### 데이터베이스
* SQLite
* MySQL
* TinyDB

## 4장, 머신러닝
### 머신러닝의 응용분야
1) Classification
	- 분류
2) Clustering
	- 그룹 나누기
3) Recommendation
4) Regression
	- 과거 데이터를 기반으로 미래 데이터 예측
5) Dimensionality Reduction
	- 데이터 축소

### [scikit-learn](https://scikit-learn.org/stable/)
- scikit-learn, scipy, matplotlib, scikit-image, pandas

#### [MNIST](http://yann.lecun.com/exdb/mnist/)
- 손글씨 숫자 데이터 제공

#### SVM (Support Vector Machine)
- 평면(공간?)에서 요소들을 구분하는 선을 찾고, 이를 기반으로 인풋값이 속하는 곳을 찾아 예측한다.

* Scikit-learn은 세 가지 종류의 SVM을 제공한다.
	1) SVC
	2) NuSVC
	3) LinearSVC

#### 랜덤 포레스트
- 집단 학습을 기반으로 고정밀 분류, 회귀, 클러스터링 등을 구현하는 것. 앙상블 기법 중 하나

#### 교차 검증 (Cross-validation)
* K 분할 교차 검증(K-fold cross validation)
* scikit-learn이 제공하는 교차 검증 [code](./ch4/ch4-7.ipynb)

#### 그리드 서치 (Grid Search)
- 인공지능 알고리즘에 어떤 파라미터 값이 적정한 지 자동으로 조사해주는 방법
* scikit-learn에서는 GridSearchCV()를 통해 제공 cf. [code](./ch4/ch4-7.ipynb)

## 5장, 딥러닝
### [ImageNet](https://www.image-net.org)
- 시각적 개체 인식 소프트웨어 연구에 사용하도록 설계된, 대규모 시각적 데이터베이스

### 합성곱 신경망 (Convolutional Neural Network, CNN)
- 입력층-(합성곱층-풀링층)-(전결합층)-출력층
- 은폐층에 합성곱층과 풀링층이 번갈아가며 반복되는 구조이다.

* 합성곱층
	- 가중치 벡터(필터)를 이용해 특징맵(c)을 만든다.
	- 필터의 종류는 다양한데, 대표적으로 평활화 (Equaliztion)과 윤곽선 검출 (Edge Detection)이 있다.
* 풀링층
	- 특징맵(c)의 크기를 축소한다.
	- 최댓값을 사용하는 최대 풀링 (max pooling)과 평균값을 사용하는 평균 풀링 (average pooling)이 있다.

#### 확률적 경사 강하법 (Stochastic Gradient Descent)
- 무작위로 초기화한 매개변수를 손실 함수가 작아지도록 지속적으로 반복해서 변경하는 것
- Tensorflow에서 모델 최적화 함수(Model Optimizer)로 'Adam'을 이용하는 것이, 이를 사용하는 것이다.

### [Keras](https://keras.io)
- Tensorflow는 고급 계산 프레임워크다.
- Keras는 Theano와 Tensorflow를 wrapping한 라이브러리이다. Tensorflow만 사용할 때보다, 비교적 쉽게 코드 작성이 가능하다.

* np_utils.to_categorical
``` python
# [5,1,9]와 같이 구성된 라벨 데이터를 [[0,0,0,0,0,1,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1]]와 같이 변경해주는 메서드
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train, 10)
```

* 배치 크기 (batch size)란?
    - 머신러닝을 수행할 때는 굉장히 많은 양의 데이터를 사용하기 때문에, 뭔가 하나를 수정한다면 모든 데이터를 다시 처리해야 할 수도 있다.
    - 따라서 훈련 데이터를 여러 개의 작은 배치로 나누어 매개변수를 수정하는데, 이를 배치 크기 (batch size) 또는 미니 배치 크기 (mini batch size)라고 한다.
    
### Pandas / Numpy [예시](./ch5/ch5-8.ipynb)
* [NumPy 매뉴얼](http://docs.scipy.org/doc/numpy)
* [Pandas 매뉴얼](http://pandas.pydata.org/pandas-docs/stable)