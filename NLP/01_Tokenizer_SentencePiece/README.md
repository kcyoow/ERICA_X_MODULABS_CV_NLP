# 01. 멋진 단어사전 만들기 — SentencePiece 토크나이저

> 회차 2 (10/29) · 텍스트데이터 다루기 · LMS 프로젝트 노드 1/4

## 파일

| 파일 | 설명 |
|---|---|
| `2 멋진 단어사전 만들기 [프로젝트].ipynb` | LMS 원본 노트북. 앞부분은 공백/형태소 토큰화 실습, 뒷부분 "프로젝트: SentencePiece 사용하기"가 제출 과제 |

## 학습 목표

- 코퍼스를 받아 길이 분포를 확인하고, 중복·노이즈를 제거하는 전처리를 할 수 있다.
- 공백 기반 → 형태소(MeCab) 기반 → 서브워드(SentencePiece) 기반 토큰화를 비교하고, 각 방식의 단어 사전 크기와 장단점을 설명할 수 있다.
- SentencePiece 모델을 직접 학습하고, 이를 분류 모델에 연결해 성능을 측정할 수 있다.

## 과제 내용

**데이터**
- 토크나이저 학습: `korean-english-park.train.ko` (한-영 병렬 말뭉치의 한국어 부분, 94,123문장 → 중복 제거·길이 필터 후 약 7.7만 문장)
- 분류 태스크: [NSMC 네이버 영화리뷰 감성 분석](https://github.com/e9t/nsmc/)

**단계**
1. SentencePiece 설치
2. 정제된 코퍼스로 SentencePiece 모델 학습 (`vocab_size=8000`, `model_type` 기본값 unigram) → `korean_spm.model`, `korean_spm.vocab`
3. `sp_tokenize(s, corpus)` 함수 작성 — 문장 리스트를 받아 패딩된 텐서와 `word_index` / `index_word` 사전을 반환
4. NSMC에 SentencePiece 적용 → 분류 모델 학습 → KoNLPy 형태소 분석기 기반 모델과 비교 → `model_type`, `vocab_size` 변경하며 성능 변화 확인

## 평가 기준

**루브릭 (3개 중 2개 이상 달성 시 만점)**

| # | 항목 | 세부 기준 |
|:---:|---|---|
| 1 | SentencePiece 모델 생성 과정 | 코퍼스 분석 → 전처리 → SentencePiece 적용 → 토크나이저 구현·동작이 빠짐없이 진행 |
| 2 | 분류 모델과 결합 | SentencePiece 토크나이저가 적용된 Text Classifier가 정상 수렴하여 **test accuracy 80% 이상** |
| 3 | 다각도 비교분석 | 다른 토크나이저(MeCab, Okt 등) 또는 SentencePiece의 다른 옵션(model_type, vocab_size)과 체계적으로 비교 |

**배점 (5점)**: 제출 충실도 2 · 루브릭 달성 2 · 수행 과정 및 해결 노력 1

## 진행 기록

**전처리**
- NSMC는 결측과 중복 리뷰를 제거해 train 146,182건을 사용했다. 리뷰 길이는 1~146자, 평균 35자.
- `korean-english-park`에 맞춰 둔 길이 필터(10~150자)는 리뷰가 훨씬 짧은 NSMC에는 맞지 않아 적용하지 않았다.
- `max_len`은 토크나이저마다 train 토큰 길이의 95% 지점으로 정했다(unigram 8,000 기준 52).
- train을 8:2로 나눠 학습 116,946건, 검증 29,236건으로 사용했다.

**구현 중 확인한 문제**
- 노트북에 주어진 `sp_tokenize` 예시는 `EncodeAsIds` 결과(list)를 그대로 `pad_sequence`에 넘기는데, `pad_sequence`는 `torch.Tensor` 리스트를 요구하므로 `torch.tensor(...)`로 감쌌다.
- SentencePiece 기본 id는 `0=<unk>, 1=<s>, 2=</s>`, pad는 비활성(-1)이다. `padding_value=0`을 쓰면 패딩과 `<unk>`가 겹치므로, 학습할 때 `--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3`으로 지정했다.

**분류 모델**
- BiLSTM(임베딩 128, hidden 128, Dropout 0.5). `pack_padded_sequence`로 패딩 구간을 건너뛴다.
- 최대 5 epoch 학습하고, validation 정확도가 가장 높았던 가중치로 test를 평가했다.

**비교 실험 결과** (Colab GPU, 조건별 1회 실행)

| 토크나이저 | 사전 크기 | 리뷰당 평균 토큰 수 | 임베딩 파라미터 | best val acc | test acc |
|---|:---:|:---:|:---:|:---:|:---:|
| MeCab 형태소 | 8,000 | 18.7 | 1,024,000 | 0.8659 | **0.8602** |
| SentencePiece unigram | 4,000 | 20.2 | 512,000 | 0.8452 | 0.8467 |
| SentencePiece unigram | 8,000 | 17.3 | 1,024,000 | 0.8435 | 0.8469 |
| SentencePiece unigram | 16,000 | 15.4 | 2,048,000 | 0.8426 | 0.8422 |
| SentencePiece bpe | 8,000 | 17.0 | 1,024,000 | 0.8433 | 0.8472 |
| MeCab 형태소 → SentencePiece | 8,000 | 20.4 | 1,024,000 | 0.8579 | 0.8568 |
| MeCab 형태소 + 품사 태그 → SentencePiece | 8,000 | 21.8 | 1,024,000 | 0.8579 | 0.8567 |

- SentencePiece(unigram 8,000)만 쓴 분류기의 test 정확도는 84.69%로 루브릭 기준 80%를 넘었다.
- `model_type`과 `vocab_size`를 바꿔도 SentencePiece 조건끼리의 차이는 0.5%p 이내였다. 사전을 4,000으로 줄이면 임베딩 파라미터는 절반이 되지만 정확도는 거의 같다.
- MeCab 형태소 기반이 86.02%로 가장 높았고, 형태소로 먼저 나눈 뒤 SentencePiece를 학습하면 SentencePiece만 쓸 때보다 약 1%p 올랐다.

## 참고 자료

- [google/sentencepiece](https://github.com/google/sentencepiece)
- [NSMC — Naver Sentiment Movie Corpus](https://github.com/e9t/nsmc/)
- [jungyeul/korean-parallel-corpora](https://github.com/jungyeul/korean-parallel-corpora)
