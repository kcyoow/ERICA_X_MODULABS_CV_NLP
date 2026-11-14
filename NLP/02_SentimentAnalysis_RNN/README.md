# 02. 영화리뷰 텍스트 감성분석하기 — RNN · 1-D CNN · Word2Vec

> 회차 3 (11/3) · 영화리뷰 텍스트 감성분석하기 · LMS 프로젝트 노드 2/4

## 파일

| 파일 | 설명 |
|---|---|
| `4 영화리뷰 텍스트 감성분석하기[프로젝트].ipynb` | 제출 과제. 네이버 영화리뷰(NSMC)를 MeCab으로 토큰화해 세 가지 모델로 감성 분류를 하고, 한국어 Word2Vec 임베딩을 적용해 비교 |

## 학습 목표

- 텍스트 데이터를 머신러닝 입출력용 수치 데이터로 변환하는 과정을 이해한다.
- RNN의 특징을 이해하고 시퀀셜한 데이터를 다루는 방법을 이해한다.
- 1-D CNN으로도 텍스트를 처리할 수 있음을 이해한다.
- IMDB와 네이버 영화리뷰 데이터셋을 이용한 영화리뷰 감성 분류 실습을 진행한다.

## 과제 내용

**데이터**
- 분류 태스크: [NSMC 네이버 영화리뷰 감성 분석](https://github.com/e9t/nsmc/) (train 15만 건, test 5만 건)
- 사전학습 임베딩: 한국어 Word2Vec ([Kyubyong/wordvectors](https://github.com/Kyubyong/wordvectors)의 `ko.bin`, 200차원, 단어 30,185개)

**단계**
1. 데이터 준비와 확인
2. 데이터로더 구성 — 중복·결측 제거, MeCab 토큰화, 불용어 제거, 상위 10,000개 단어 사전, 인덱스 변환
3. 문장 길이 분포를 보고 최대 길이를 정한 뒤 패딩
4. 모델 3가지 이상 구성, validation set 분리
5. 모델 훈련
6. Loss, Accuracy 그래프 시각화
7. 학습된 Embedding 레이어 분석 (gensim 유사 단어 찾기)
8. 한국어 Word2Vec 임베딩을 활용해 성능 개선

## 평가 기준

**루브릭 (3개 중 2개 이상 달성 시 만점)**

| # | 항목 | 세부 기준 |
|:---:|---|---|
| 1 | 한국어 Word2Vec을 활용하여 가시적인 성능 향상 달성 | 네이버 영화리뷰 데이터 감성분석 **정확도 85% 이상** |
| 2 | gensim을 활용한 임베딩 레이어 분석 | gensim의 유사 단어 찾기로 자체학습한 임베딩과 사전학습 임베딩을 비교 분석 |
| 3 | 다양한 방법으로 Text Classification 구현 | 3가지 이상의 모델이 성공적으로 시도됨 |

**배점 (5점)**: 제출 충실도 2 · 루브릭 달성 2 · 수행 과정 및 해결 노력 1

## 진행 기록

**전처리**
- 중복·결측을 제거한 train 146,182건을 사용했다. 토큰 수는 평균 16.0개, 최대 116개.
- 최대 길이는 평균 + 2 × 표준편차인 41로 정했다. 전체 문장의 93.4%가 이 길이 안에 들어온다.
- train의 앞 10,000건을 validation으로 떼고 136,182건으로 학습했다.

**구현 중 확인한 문제**
- **패딩 방향**: 강의 코드의 패딩은 문장 뒤에 0을 채운다(post). LSTM은 마지막 state만 쓰는데, 평균 16개 토큰 뒤에 0을 25개쯤 더 읽고 난 state가 되어 1 epoch 학습 후 test 정확도가 0.50에 머물렀다. 문장 앞에 채우도록(pre) 바꾸자 같은 조건에서 0.72가 되어 pre 패딩을 사용했다.
- **사전학습 Word2Vec 파일**: 과제 안내의 `word2vec_ko.model`이 수업 자료에 없어 공개된 한국어 Word2Vec(`ko.bin`)을 받았다. 이 파일은 오래된 gensim 형식이라 gensim 4.x에서 열리지 않아, 단어와 벡터만 꺼내 gensim 4.x의 Word2Vec 형식으로 다시 저장해 사용했다.
- **MeCab 설치 순서**: Colab에서 `python-mecab-ko`는 소스 빌드가 필요해 `mecab-config`가 있어야 한다. MeCab 설치 스크립트를 먼저 실행해야 설치된다.
- **같은 코드의 반복**: 강의의 학습·평가·그래프 코드를 모델마다 복사하는 대신 `ModelTrainer` 클래스로, 임베딩 저장·읽기·유사 단어 찾기를 `EmbeddingAnalyzer` 클래스로 묶어 네 모델에 같은 코드를 썼다.

**모델과 학습 조건**
- LSTM(state 8), 1-D CNN(채널 16, kernel 7), GlobalMaxPooling. 임베딩 16차원, 구조와 크기는 강의 예시 그대로.
- Word2Vec 모델은 같은 1-D CNN 구조에 임베딩만 사전학습 Word2Vec(200차원)으로 채우고 함께 학습했다.
- Adam, batch 512, 5 epoch. Colab CPU에서 모델별 1회 실행, 시드 고정 없음.

**모델별 결과**

| 모델 | 임베딩 | val acc (5 epoch) | test acc |
|---|---|:---:|:---:|
| LSTM | 자체 학습 16차원 | 0.8280 | 0.8259 |
| 1-D CNN | 자체 학습 16차원 | 0.8283 | 0.8215 |
| GlobalMaxPooling | 자체 학습 16차원 | 0.7705 | 0.7701 |
| 1-D CNN + Word2Vec | 사전학습 200차원 | 0.8515 | **0.8507** |

- 같은 1-D CNN 구조에서 임베딩만 사전학습 Word2Vec으로 바꾸자 test 정확도가 82.15% → 85.07%로 약 2.9%p 올랐다.
- 85.07%는 기준 85%를 0.07%p 넘긴 1회 실행 결과다. 시드를 고정하지 않았으므로 다시 실행하면 값이 조금 달라질 수 있다.
- Word2Vec 모델의 validation loss는 4 epoch에서 가장 낮고 5 epoch에서 다시 올랐다.

**임베딩 비교** (gensim `similar_by_word`, 상위 5개)

| 단어 | 자체 학습 임베딩 (LSTM) | 사전학습 Word2Vec |
|---|---|---|
| 끝 | 왜, 공포감, 찾, 지식, 남긴다 | 겨드랑이, 마지막, 발목, 밑부분, 허벅지 |
| 재미 | 양아치, 유태인, 테론, 어선, MBC | 취미, 유머, 매력, 흥미, 공짜 |
| 감동 | 오즈, 아름다워요, 키는, 아낌없, 조타 | 감명, 감격, 실망, 감화, 감탄 |

- 감성 분류만 5 epoch 학습한 16차원 임베딩은 세 모델 모두 뜻이 가까운 단어를 찾지 못했다(CNN, GlobalMaxPooling의 결과는 노트북 7-3 참고).
- 위키백과로 학습한 사전학습 Word2Vec은 뜻이 가까운 단어를 찾는다. 다만 `감동`의 이웃에 `실망`이 있는 것처럼, 같은 자리에 쓰이는 반대 감정의 단어도 가깝게 나온다.

## 참고 자료

- [NSMC — Naver Sentiment Movie Corpus](https://github.com/e9t/nsmc/)
- [Kyubyong/wordvectors — Pre-trained word vectors of 30+ languages](https://github.com/Kyubyong/wordvectors)
- [SOMJANG/Mecab-ko-for-Google-Colab](https://github.com/SOMJANG/Mecab-ko-for-Google-Colab)
- [gensim KeyedVectors 문서](https://radimrehurek.com/gensim/models/keyedvectors.html)
- Dwarampudi, M. & Reddy, N. V. S. *Effects of padding on LSTMs and CNNs.* arXiv:1903.07288 (2019). https://arxiv.org/abs/1903.07288
