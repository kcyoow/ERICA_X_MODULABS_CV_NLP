# ERICA_X_MODULABS_CV_NLP

한양대학교 ERICA × 모두의연구소 IC-PBL 과정에서 진행한 **자연어처리(NLP)** 와 **컴퓨터비전(CV)** 실습 프로젝트를 정리한 레포지토리입니다.

- 각 실습 프로젝트는 번호가 붙은 폴더 하나로 구성되며, 폴더마다 실습 노트북(`.ipynb`)과 해당 프로젝트의 설명·진행 기록을 담은 `README.md`가 들어 있습니다.
- 폴더 번호와 이름은 과정별 시간표를 따릅니다. 시간표가 바뀌면 구조도 함께 갱신합니다.

## 프로젝트 목록

| 폴더 | 회차 | 주제 | 과제 노트북 | 주요 결과 (NSMC test 정확도) |
|---|:---:|---|---|---|
| [NLP/01_Tokenizer_SentencePiece](NLP/01_Tokenizer_SentencePiece) | 2 (10/29) | 텍스트데이터 다루기 — SentencePiece 토크나이저 | `2 멋진 단어사전 만들기 [프로젝트].ipynb` | SentencePiece + BiLSTM 84.69%, MeCab 형태소 86.02% |
| [NLP/02_SentimentAnalysis_RNN](NLP/02_SentimentAnalysis_RNN) | 3 (11/3) | 영화리뷰 텍스트 감성분석하기 — RNN · 1-D CNN · Word2Vec | `4 영화리뷰 텍스트 감성분석하기[프로젝트].ipynb` | LSTM 82.59%, 1-D CNN + 한국어 Word2Vec 85.07% |

## 디렉토리 구조

```bash
ERICA_X_MODULABS_CV_NLP
├── README.md
└── NLP
    ├── 01_Tokenizer_SentencePiece
    │   ├── 2 멋진 단어사전 만들기 [프로젝트].ipynb
    │   └── README.md
    └── 02_SentimentAnalysis_RNN
        ├── 4 영화리뷰 텍스트 감성분석하기[프로젝트].ipynb
        └── README.md
```
