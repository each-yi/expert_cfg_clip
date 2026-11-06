# 재현 조사 기록

## PubMedCLIP (Expert-CFG Table 1) → QCR + AE

PubMedCLIP 원 논문 Table 1(p.1186)에 따르면 VQA-RAD 60.1/80.0/72.1은 **RN50 + AE**, SLAKE 78.4/82.5/80.1은 **ViT-B/32 + AE** 결과입니다.
공식 QCR 코드 commit `bdecf3c5a21b68200a02dc05455b3882a23206a2` 기준 설정은 다음과 같습니다.
- GloVe+GRU 1024, 질문 길이 12, TF-IDF + CAT 임베딩
- Open/Closed별 BAN(glimpse 10), 사전학습 질문 유형 분류기(TCR)
- 사전학습 AE + 재구성 손실 가중치 0.001, BCEWithLogits
- Adamax, lr 1e-3, batch 16, 200 epoch, seed 88. 논문 본문에는 Adam으로 적혀 있지만 코드는 Adamax이며, 재현은 코드를 따릅니다.
- 매 epoch test로 평가하고 최고 epoch를 보고합니다(저자 코드). 이전 조사 기록에 따르면 논문은 10회 평균입니다.

### VQA-RAD 데이터를 원래 QCR 묶음에서 가져오는 이유 (2026-11-06 확인)

PubMedCLIP 저장소 `lib/utils/run.sh`의 스크립트로 RAD를 다시 만들면 공식 학습 코드가 그대로 돌지 않습니다.

| 항목 | 공식 코드가 요구하는 값 | PubMedCLIP 스크립트로 다시 만든 값 | 원래 QCR `data/` |
|---|---|---|---|
| 라벨 공간 (`dataset_RAD.py` 하드코딩) | 487 = closed 56 + open 431, open 라벨은 0부터 | 64 + 483, open 라벨에 closed 개수가 더해짐 | 56 + 431, open 최대 430 |
| 사전 (`saved_models/type_classifier.pth` 임베딩 1178행) | 1177단어 | 1259단어 → 로드 실패 | 1177단어 |

- 원래 QCR 저장소 [Awenbocc/med-vqa](https://github.com/Awenbocc/med-vqa) @ `0cca6811`의 `data/`가 위 요구값과 정확히 맞습니다.
  이 저장소의 `type_classifier.pth`는 PubMedCLIP 저장소의 것과 SHA-256까지 같습니다.
- 그 묶음의 `trainset.json`(3064), `testset.json`(451, Open 179 / Closed 272), `imgid2idx.json`, `pretrained_ae.pth`, `images128x128.pkl`,
  `dictionary.pkl`은 HF `aiozai/MEVF` `data_RAD.zip`과 바이트 단위로 같습니다. 즉 MEVF 표준 split입니다.
- 다른 파일은 `embed_tfidf_weights.pkl`과 cache뿐이며, 원래 QCR 것을 씁니다. 공식 코드의 `main/language_model.py`는 이 pickle을 불러오기 위해 있습니다.
- 묶음에 없는 `images250x250.pkl`(CLIP 입력)만 PubMedCLIP `create_resized_images.py`로 만듭니다. 결과는 결정적입니다(재생성 시 같은 해시).
- HF에서 원래 QCR cache를 찾으려고 VQA-RAD/SLAKE/MedVQA 관련 저장소 653개의 파일 목록을 검색했지만 없었습니다.

### SLAKE

- HF `BoKelvin/SLAKE`의 train/test JSON에서 영어만 사용합니다(4919/1061). PubMedCLIP `run.sh`의 SLAKE 단계를 그대로 실행합니다.
- 다시 만든 사전은 292단어로, `type_classifier_slake.pth` 임베딩 293행과 일치합니다. 라벨은 close 35 / open 219입니다.
- QCR은 SLAKE용 AE를 제공하지 않습니다. config가 `data_slake/pretrained_ae.pth`를 기대하므로 MEVF AE(RAD와 같은 파일)를 둡니다. 이 부분은 저자 설정을 확인할 수 없는 가정입니다.
- 공식 코드는 SLAKE validation을 쓰지 않습니다.

## BioMedCLIP — 동일 재현 불가 → 근사 실험

2026-11-06에 arXiv 2303.00915v1 VQA 부분을 다시 확인했습니다.
- "We utilize the METER framework … formulates the VQA task as a classification task."
  같은 절에 "All three models were fine-tuned … using the QCR framework"라는 모순된 문장도 있지만, Table 11은 BiomedCLIP 행만 METER로 표기합니다.
- 모델명은 "ViT-B/16-448-PMB/256"뿐입니다. VQA 해상도, fusion 층 수, epoch/lr/batch/loss/답 사전/모델 선택/실행 횟수는 **적혀 있지 않습니다**.
  Appendix A의 설정(lr 5e-4, 40 epoch 등)은 사전학습용입니다.
- Table 7의 VQA-RAD 3064/451은 QCR 재현과 같은 split입니다.
  SLAKE는 본문에 "English only"라고 쓰여 있지만 Table 7의 수치(9849/2109/2070)는 두 언어를 합친 규모라 서로 맞지 않습니다. 근사 실험은 본문을 따라 영어만 씁니다.
- 공개된 것은 사전학습 모델(aka.ms/biomedclip)뿐입니다. HF에는 224 체크포인트(`9f341de`)만 있고, 비슷한 이름의 사본들도 같은 224 모델을 복사한 것입니다.
  공식 GitHub(`microsoft/BiomedCLIP_data_pipeline`)에는 사전학습 데이터 처리 코드만 있습니다. 일반 METER 저장소는 2022년 이후 변경이 없습니다.

빈 설정은 METER 기반 의료 VQA 공식 구현인 M3AE @ `14da377`에서 가져왔습니다.
- `task_finetune_vqa_vqa_rad`: batch 64, 50 epoch, lr 5e-6, head ×50, multimodal ×5, warmup 0.1, max_text_len 32, `vqa_label_size=498`
- `task_finetune_vqa_slack`: batch 32, 15 epoch, `vqa_label_size=222`
- 공통 설정: AdamW, wd 0.01, linear decay, seed 0
- 답 사전은 모든 split 답을 `normalize_word`로 정규화해 만들고, 타깃 점수는 1.0, 손실은 BCE × 라벨 수입니다. 평가는 최고 epoch를 보고합니다.

우리 정규화 구현으로 만든 라벨 수는 RAD 498, SLAKE 222로 M3AE와 정확히 같습니다.

M3AE와 다른 점:
- 인코더가 BiomedCLIP 224입니다(M3AE는 CLIP ViT-B/32를 576으로 학습).
- M3AE의 MIM/MLM 사전학습 fusion 가중치는 쓰지 않습니다. fusion은 랜덤 초기화합니다.

PubMedCLIP QCR 결과와 비교할 때 주의할 점 (2026-11-07 확인):
- QCR `evaluate_classifier`는 `forward_classify` 안에서 질문 유형 분류기의 **예측**으로 open/closed를 나눕니다.
  그래서 로그의 VQA-RAD test 분모는 180/271이고, `testset.json`의 정답 `answer_type`은 179/272입니다.
- BiomedCLIP 근사는 정답 `answer_type`으로 나눕니다(M3AE 방식). 따라서 두 모델 간 Open/Closed는 분모가 다르고, Overall(451)만 같은 기준입니다.

## BioMed-VITAL (13B, 60K) — 공식 fine-tuning 코드 없음 → 근사 재현 (2026-11-07 조사)

목표 수치(Expert-CFG Table 1 = BioMed-VITAL 논문 Table 4): VQA-RAD 64.88 / 84.55 / 76.74, SLAKE 87.82 / 86.54 / 87.32 (Open / Closed / Overall).

공개된 것과 공개되지 않은 것:
- 공식 저장소 [mao1207/BioMed-VITAL](https://github.com/mao1207/BioMed-VITAL) @ `f6ccf75`에는 데이터 생성(`generate/`), 데이터 선택(`filter/`), 추론(`eval/model_vqa_med.py`), 지표 함수(`eval/evaluate_metrics.py`)만 있습니다.
  VQA fine-tuning 스크립트, 학습 명령, 설치 방법은 없습니다. 논문 부록 H.6은 "All the prompts, instructions, and model checkpoints for reproducing the results can be found in the GitHub repository"라고 적고 있어 실제 저장소와 맞지 않습니다.
- 가중치는 HF `mao1207/BioMed-VITAL-models` @ `c93f260` 하나뿐입니다.
  LLaVA 13B 구조, CLIP ViT-L/14-336, `mlp2x_gelu`, fp32 약 52 GB입니다. 60K/150K 표시가 없고, `trainer_state.json`은 epoch 6.0이라 논문의 instruction tuning 3 epoch과 맞지 않습니다.
- 논문(arXiv 2406.13173 v1/v3) 4.4절의 fine-tuning 설정은 한 문장뿐입니다. "For each benchmark, the model is fine-tuned for 15 epochs with a learning rate of 2e-5."
  4.1절은 "Following the LLaVA-Med approach"라고 적고, 백본은 llava-v1.5-13b입니다.
- 평가: closed는 정확도, open은 recall(정답 토큰이 생성문에 나타나는 비율)입니다. SLAKE는 영어만 씁니다. split은 Table 3 기준 VQA-RAD 1,797/451, SLAKE 4,919/1,061입니다.

논문에 빠진 값은 BioMed-VITAL이 따랐다고 밝힌 LLaVA-Med v1의 downstream fine-tuning 명령으로 채웁니다.
출처는 [microsoft/LLaVA-Med](https://github.com/microsoft/LLaVA-Med/tree/v1.0.0) README, tag `v1.0.0` = `b9a98a7`의 "Medical VQA > Fine-tuning" 절입니다.
- "the same full-model tuning script with instruct tuning is used to continue train LLaVA-Med", 즉 전체 모델 fine-tuning입니다. vision encoder는 고정합니다(LLaVA-Med 논문 Stage 2).
- 8 GPU × per-device batch 1 × gradient accumulation 8 = 전역 batch 64
- lr 2e-5(BioMed-VITAL 논문 값과 같음), weight decay 0, warmup 0.03, cosine, bf16, `model_max_length 2048`, gradient checkpointing
- 명령의 `num_train_epochs`는 3이지만, 공개된 fine-tuned delta 이름은 `data_RAD-9epoch`, `Slake1.0-9epoch`입니다. 우리는 BioMed-VITAL 논문 값 15를 씁니다.
- LLaVA-Med 평가 코드(`model_vqa_med.py`)는 `do_sample=True, temperature=0.7`로 샘플링합니다. BioMed-VITAL 저장소의 `eval/model_vqa_med.py`도 같습니다.
  `run_eval.py`에서 closed는 "pred에 yes/no가 있으면 gt in pred", open은 `calculate_appearance_with_normalization`이고, 별도로 token recall을 출력합니다.
- VQA-RAD/SLAKE train/test JSON은 LLaVA-Med에서도 공개되지 않았습니다.

참고: BioMed-VITAL Table 4의 LLaVA-Med 13B 행(64.58/77.94/84.97/85.58/38.82/92.39)은 LLaVA-Med 논문 Table 4(b)의 13B, FT 9 epoch 행과 같습니다.
7B 행(61.52/84.19/…)은 LLaVA-Med의 "From LLaVA" 행과 같습니다. 즉 LLaVA-Med 수치는 다시 돌린 값이 아니라 인용한 값입니다.

## 출처

- [Expert-CFG Table 1](https://arxiv.org/html/2507.09209v1)
- [PubMedCLIP 원 논문](https://aclanthology.org/2023.findings-eacl.88.pdf), [공식 QCR 코드](https://github.com/sarahESL/PubMedCLIP/tree/bdecf3c5a21b68200a02dc05455b3882a23206a2/QCR_PubMedCLIP)
- [원래 QCR 코드·데이터](https://github.com/Awenbocc/med-vqa/tree/0cca6811e38cf54aff6a7cce3442296d07875e64)
- [MEVF VQA-RAD (HF)](https://huggingface.co/datasets/aiozai/MEVF), [PubMedCLIP 가중치 (HF)](https://huggingface.co/sarahESL/PubMedCLIP), [SLAKE (HF)](https://huggingface.co/datasets/BoKelvin/SLAKE)
- [BioMedCLIP](https://arxiv.org/html/2303.00915v1)
