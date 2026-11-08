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

> 실험 진행하면서 채워 넣는 섹션. 선택 근거, 막힌 지점과 해결, 결과를 순서대로 남긴다.

**전처리**
- (작성 예정) NSMC 리뷰 길이 분포 / max_len 결정 근거

**구현 중 확인한 문제**
- 노트북에 주어진 `sp_tokenize` 예시는 `EncodeAsIds` 결과(list)를 그대로 `pad_sequence`에 넘기는데, `pad_sequence`는 `torch.Tensor` 리스트를 요구하므로 `torch.tensor(...)`로 감싸야 한다.
- SentencePiece 기본 id는 `0=<unk>, 1=<s>, 2=</s>`, pad는 비활성(-1). `padding_value=0`을 쓰면 패딩과 `<unk>`가 겹치므로, 학습 시 `--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3`으로 지정하거나 `nn.Embedding(padding_idx=...)`를 따로 맞춘다.
- `korean-english-park`에 맞춰 둔 길이 필터(10~150자)는 NSMC에는 맞지 않음(리뷰가 훨씬 짧음). NSMC 분포를 다시 보고 기준을 잡아야 한다.

**비교 실험 결과**

| 토크나이저 | model_type | vocab_size | 문장당 평균 토큰 수 | 임베딩 파라미터 | test acc |
|---|---|:---:|:---:|:---:|:---:|
| MeCab | – | – | | | |
| SentencePiece | unigram | 4,000 | | | |
| SentencePiece | unigram | 8,000 | | | |
| SentencePiece | unigram | 16,000 | | | |
| SentencePiece | bpe | 8,000 | | | |

---

## 심화: 임베딩 테이블 오버헤드와 Kronecker Embeddings

### 왜 이 주제인가

토크나이저의 vocab 크기 `V`는 토크나이저 파일 크기보다 **모델 안의 두 행렬**에 영향을 준다.

- 입력 임베딩 `V × d`
- 출력 LM head `d × V`

몸통(어텐션·FFN)이 작아질수록 이 두 행렬의 비중이 커진다. 공개된 수치로는:

| 모델 | vocab | 임베딩(+head) 비중 | 출처 |
|---|:---:|:---:|---|
| 125M (d=512, V=32k) | 32k | 20% 이상 | MobileLLM |
| LLaMA-7B / 70B | 32k | 3.7% / 0.7% | MobileLLM |
| Gemma 2 2.6B (tied) | 256k | 22% | Modular Tokenizers (EMNLP 2026) |
| Llama 3.2 1B | 128k | 메모리의 20% 이상 | VocabTailor (ACL 2026) |
| Gemma 3 270M | 262k | **63%** (268M 중 168M) | Embeddings Are Overpriced? |

이번 과제의 `vocab_size` 비교 실험은 이 트레이드오프를 소형 분류기에서 직접 측정해 보는 것과 같다. 여기서 한 걸음 더 나가, **임베딩 테이블 자체를 없애는 방법**을 하나 골라 같은 분류기에서 재현해 본다.

### 논문: Kronecker Embeddings (Shravan, arXiv 2605.29459, 2026.05)

**문제** — 모든 트랜스포머 LM은 토큰 id → `|V| × d_model` 학습 테이블 lookup으로 시작한다. 이 테이블은 소형 모델에서 파라미터의 1/4~절반을 차지하고, 양자화 시에도 보통 16비트로 남는다.

**방법** — 학습되는 테이블을 **고정 인코더 + 학습되는 projection 하나**로 교체한다.

1. 토큰의 UTF-8 바이트열 `b_1 … b_L`에 대해, 각 위치 `p`에서 (바이트 값 one-hot `c_{b_p}` ∈ R^{d_c}) ⊗ (위치 one-hot `p_p` ∈ R^{d_p})를 계산한다. 결과는 `D = d_c · d_p` 차원에서 좌표 `b_p · d_p + p` 하나만 1인 벡터.
2. 토큰의 모든 바이트에 대해 이를 합산하고 길이로 정규화하면, 토큰마다 **결정론적인 sparse multi-hot 벡터** `κ ∈ R^D`가 나온다. 학습 파라미터 없음, gradient 없음. 논문 설정은 `d_c = 256, d_p = 16, D = 4096`.
3. 학습되는 선형층 `W: R^D → R^{d_model}` 하나로 projection. 입력 쪽 학습 파라미터는 `D × d_model`이며 **vocab 크기와 무관**하다.

ALBERT의 factorized embedding(`E = V_E · E_F`)과 같은 계열이지만, 첫 번째 인자가 학습되지 않고 토큰의 바이트 구성으로 결정된다는 점이 다르다.

**결과 (논문 보고 기준)**
- nanoGPT GPT-2 124M, FineWeb-Edu 2.5B 토큰, 3시드: 입력 임베딩 38.6M → 3.1M으로 줄이면서 validation loss **2.5 ± 0.2% 감소**(약 9% 낮은 perplexity), 18/18 체크포인트×시드에서 일관.
- 오타 1글자 섞은 110쌍 프롬프트에서 top-1 예측 유지율 55.5% vs 47.3% — 바이트 유사성이 임베딩 유사성으로 이어지는 "byte-level locality" 덕분.
- 프로덕션 규모(vocab 131k, D=8192): 2.15GB 테이블 대신 4.5MB 바이트 버퍼 + projection으로 대체, 재계산 오버헤드 0.01~0.24%.

**논문이 밝힌 한계**
- 바이트가 비슷하면 의미가 달라도 가까워진다 (`compute`/`commute`). 구분은 첫 어텐션 층의 몫.
- tied embedding 모델에서는 output head가 분리되므로 124M 기준 **전체 파라미터는 +3.1M**. "91~94% 절감"은 입력 쪽만의 수치다.
- 단독 저자 preprint이며, CharCNN·ByT5 같은 바이트 수준 베이스라인과의 직접 비교가 빠져 있다.

### 이 과제에서의 재현 설계

분류기에는 LM head가 없으므로 위의 "+3.1M" 문제가 없고, 입력 쪽 절감이 그대로 전체 절감이 된다.

**고정**: SentencePiece(unigram) 토크나이저, BiLSTM(hidden 128) + Linear 분류기, 동일한 학습 설정. **임베딩 층만 교체.**

| 조건 | 입력 임베딩 | 학습 파라미터 (V=8k) | (V=32k) |
|---|---|:---:|:---:|
| Baseline | `nn.Embedding(V, 128)` | 1.02M | 4.1M |
| Kron-byte | 바이트 × 위치 (256 × 16 = 4096) → `Linear(4096, 128)` | 0.52M | 0.52M |
| Kron-jamo | 초·중·종성 자모 × 위치 (~70 × 8 = 560) → `Linear(560, 128)` | 0.07M | 0.07M |

- **Kron-jamo**는 한국어 특화 변형. UTF-8에서 한글 한 글자는 3바이트라 바이트 단위는 낭비가 크다. 초성 19 · 중성 21 · 종성 28(없음 포함)로 분해하면 알파벳이 약 70개로 줄어 `D`가 7배 작아진다. 논문도 바이트 bigram 등 다른 factorization이 가능하다고 열어 두었다.
- 구현: `[V, D]` codec 테이블을 미리 한 번 계산해 두고 forward는 `K[ids] @ W`. 자모 버전은 32k × 560 float32 ≈ 72MB라 dense로 충분하다.

**측정 항목**
1. test accuracy — vocab 4k / 8k / 16k / 32k × 3조건
2. 임베딩 쪽 학습 파라미터 수 (실측)
3. 오타 강건성 — 테스트 리뷰에서 한 글자씩 치환한 뒤 accuracy 재측정 (논문의 typo probe를 분류 버전으로)
4. 수렴 속도 — epoch별 validation accuracy 곡선

**예상과 리스크**
- 분류 태스크는 LM보다 임베딩 표현력 의존도가 낮아 accuracy는 비슷하거나 소폭 하락할 가능성이 크다. 파라미터 절감 대비 하락 폭, 그리고 오타 강건성 변화를 함께 보는 것이 목적이다.
- 결과가 나쁘더라도 "LM에서의 이득이 분류로 전이되지 않는 이유"를 분석하면 그 자체가 회고가 된다. 결과 방향을 미리 정하지 않는다.

**결과** (작성 예정)

| 조건 | vocab | 임베딩 파라미터 | test acc | 오타 acc | 수렴 epoch |
|---|:---:|:---:|:---:|:---:|:---:|
| Baseline | 8k | | | | |
| Kron-byte | 8k | | | | |
| Kron-jamo | 8k | | | | |
| Baseline | 32k | | | | |
| Kron-byte | 32k | | | | |
| Kron-jamo | 32k | | | | |

## 회고

(작성 예정)

## 참고 자료

- [google/sentencepiece](https://github.com/google/sentencepiece)
- [NSMC — Naver Sentiment Movie Corpus](https://github.com/e9t/nsmc/)
- [jungyeul/korean-parallel-corpora](https://github.com/jungyeul/korean-parallel-corpora)
- Shravan, R. *Kronecker Embeddings: Byte-Level Structured Token Representations for Parameter-Efficient Language Models.* arXiv:2605.29459 (2026). https://arxiv.org/abs/2605.29459
- Liu, Z. et al. *MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases.* arXiv:2402.14905 (2024). https://arxiv.org/abs/2402.14905
- *To Each Language Its Tokenizer: Modular Tokenizers for Efficient Multilingual LLMs.* Findings of EMNLP 2026. https://aclanthology.org/2026.findings-emnlp.1019.pdf
- *VocabTailor: Dynamic Vocabulary Selection for Downstream Tasks in Small Language Models.* Findings of ACL 2026. https://aclanthology.org/2026.findings-acl.1418.pdf
- Nozaki, Y. et al. *Efficient Vocabulary Reduction for Small Language Models.* COLING 2025 Industry Track. https://aclanthology.org/2025.coling-industry.64.pdf
- Bathula, S. K. *Embeddings Are Overpriced? A Frozen 16 MB Vocabulary for Small and Ternary Language Models.* arXiv:2610.22428 (2026). https://arxiv.org/abs/2610.22428
- Lan, Z. et al. *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations.* ICLR 2020.
