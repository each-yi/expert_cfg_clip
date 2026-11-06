# Expert-CFG Table 1 baseline: PubMedCLIP 재현 · BiomedCLIP 근사

| 노트북 | 내용 | 성격 |
|---|---|---|
| **[PubMedCLIP_QCR_Reproduction.ipynb](notebooks/PubMedCLIP_QCR_Reproduction.ipynb)** | 공식 QCR 코드 + 저자 가중치·config | **재현** |
| **[BiomedCLIP_METER_Approximation.ipynb](notebooks/BiomedCLIP_METER_Approximation.ipynb)** | 공개 224 가중치 + METER 재구현 + M3AE 설정 | **근사**(동일 재현 불가) |

각 노트북 하나만 Colab에 업로드하고 GPU 런타임에서 위부터 실행합니다. 필요한 코드는 노트북 안에 다 들어 있습니다. 두 노트북은 같은 VQA-RAD/SLAKE 파일을 씁니다.

# PubMedCLIP (QCR) 재현

| Dataset | 인코더 | 논문 Open / Closed / Overall |
|---|---|---|
| VQA-RAD (3064 / 451) | PubMedCLIP RN50 + AE | 60.1 / 80.0 / 72.1 |
| SLAKE 영어 (4919 / 1061) | PubMedCLIP ViT-B/32 + AE | 78.4 / 82.5 / 80.1 |

## 무엇을 실행하나

- 공식 [sarahESL/PubMedCLIP `QCR_PubMedCLIP`](https://github.com/sarahESL/PubMedCLIP/tree/bdecf3c5a21b68200a02dc05455b3882a23206a2/QCR_PubMedCLIP)
  commit `bdecf3c`를 clone합니다. 저자의 config(`qcr_pubmedclipRN50_ae_rad_…`, `qcr_pubmedclipViT_ae_slake_…`)로 `main/main.py`를 실행합니다.
- config에서 바꾸는 것은 `CLIP_PATH`(다운로드한 가중치 경로)와 실행 이름뿐입니다.
- 코드 변경은 최신 Colab(PyTorch 2.x, pandas 2, Python 3.12)에서 돌리기 위한 호환성 패치뿐입니다(`qcr_port.PATCHES`).
  - `torch._six`, `collections.Mapping`, `DataFrame.append`, 구식 sparse 생성자 → 현재 API로 교체
  - `Dropout(inplace=True)` → `inplace=False`: 현재 autograd가 거부하는 연산이며, 순전파와 기울기는 같습니다.
  - `embed_tfidf_weights.pkl`(PyTorch 0.4 pickle): 검사한 클래스만 허용하는 Unpickler로 로드합니다.
  - `clip.load("RN50")`가 곧바로 덮어쓸 OpenAI 가중치를 먼저 받는 대신, PubMedCLIP state_dict로 같은 구조를 만듭니다.
    같은 fp16 변환 경로라 가중치가 동일합니다.
  - `set_detect_anomaly(True)`(디버깅용, 수치 영향 없음, 매우 느림)는 기본으로 끕니다.
  - test 점수가 오를 때마다 저장되는 약 1GB checkpoint는 최신 1개만 남깁니다(디스크 용량 때문).
- 평가 방식은 저자 코드 그대로입니다. 매 epoch **test**로 평가하고 최고 overall epoch를 보고하며, 마지막 epoch 점수도 함께 기록합니다.

## 데이터와 가중치 (모두 revision/commit 고정 + SHA-256 검사)

| 자료 | 출처 |
|---|---|
| PubMedCLIP RN50 / ViT-B/32 | HF `sarahESL/PubMedCLIP` @ `c5885a5` |
| VQA-RAD 3064/451 묶음 (cache, dictionary, TF-IDF, AE, 이미지) | 원래 QCR 저자 GitHub `Awenbocc/med-vqa` @ `0cca681` `data/` |
| SLAKE train/test JSON, 이미지 | HF `BoKelvin/SLAKE` @ `a9083ce` |
| GloVe 6B-300d, AE (SLAKE용) | HF `aiozai/MEVF` @ `7b3bd1b` `data_RAD.zip` |

VQA-RAD를 원래 QCR 묶음에서 가져오는 이유는 [PAPER_REPRODUCTION.md](PAPER_REPRODUCTION.md)에 정리했습니다.
요약하면, PubMedCLIP 저장소의 전처리 스크립트로 RAD를 다시 만들면 공식 학습 코드가 그 결과를 불러올 수 없습니다.

## 파일

- `qcr_port.py`: 공식 코드 clone, 패치, 데이터 준비, 학습 실행, 로그 파싱, 요약
- `build_qcr_notebook.py`: `qcr_port.py`를 넣어 노트북을 다시 생성합니다. `qcr_port.py`를 고친 뒤 `python build_qcr_notebook.py`를 실행하세요.
- `tests/test_qcr_port.py`: 패치·config·로그 파싱 검사(GPU와 다운로드 불필요). 실행: `python tests/test_qcr_port.py`
- `tests/qcr_cpu_smoke.py`: 준비된 데이터로 공식 `train()`을 작은 부분집합에서 CPU로 1 epoch 실행
- `upstream/PubMedCLIP`: 공식 코드 로컬 사본(git 제외, 테스트용)

## 검증 상태

로컬 CPU에서 확인한 것:
- 두 데이터셋 모두 준비 완료. RAD는 3064/451, close 56 / open 431 / 487, 사전 1177. SLAKE는 4919/1061, close 35 / open 219, 사전 292.
  두 사전 크기 모두 동봉된 type classifier와 일치합니다.
- 패치된 공식 `train()`이 RN50(RAD)과 ViT-B/32(SLAKE) 모두에서 부분집합 1 epoch 학습과 test 평가를 마칩니다.

**GPU에서 200 epoch 전체를 돌린 결과와 논문 수치 일치 여부는 아직 확인하지 않았습니다.**

# BiomedCLIP (METER) 근사

BiomedCLIP 논문 Table 11의 수치(VQA-RAD 67.6/79.8/75.2, SLAKE 82.5/89.7/85.4)는 비공개 **ViT-B/16-448** 모델로 나온 값입니다.
VQA 코드·학습 설정도 공개되지 않았습니다. 논문에는 "METER framework"라는 언급과 데이터셋 정보만 있습니다.
그래서 그대로 재현할 수 없고, 공개된 자료만으로 최대한 가깝게 구성했습니다.

- **인코더**: HF `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` @ `9f341de`(SHA-256 검사).
  open_clip 없이 timm ViT-B/16과 `BertModel`에 직접 불러옵니다(키 일치를 엄격하게 검사). 전체 fine-tuning합니다.
- **Fusion**: METER 구조를 재구현했습니다. modality별 co-attention 6층(self→cross→FFN), CLS pooler, VQA head이며, 새 모듈은 METER 방식으로 초기화합니다.
- **설정**: METER 기반 의료 VQA 공식 config인 M3AE(`zhjohnchan/M3AE` @ `14da377`)를 따릅니다.
  - AdamW lr 5e-6, cross-modal ×5, head ×50, wd 0.01, warmup 10% + linear decay, BCE × 라벨 수
  - 질문 32 token, 증강 없음
  - RAD batch 64·50 epoch, SLAKE batch 32·15 epoch
- **답 사전**: 모든 split의 정규화된 답입니다. RAD 498개, SLAKE 222개로 M3AE config 값과 일치합니다.
- **평가**: 매 epoch test로 평가해 최고 epoch를 보고합니다(QCR·M3AE 방식). 마지막 epoch와 SLAKE validation 선택 epoch도 함께 보고합니다.
  epoch마다 train accuracy(dropout이 켜진 학습 모드 기준, QCR `Train_Acc`와 같은 방식)도 기록합니다.
- **Open/Closed 기준**: 정답 `answer_type`으로 나눕니다(RAD test 179/272). PubMedCLIP QCR은 질문 유형 분류기의 예측으로 나누므로(180/271),
  두 모델 사이에서 분모가 같은 지표는 Overall뿐입니다.
- **해상도**: `IMAGE_SIZE=448`로 바꾸면 METER처럼 위치 임베딩을 보간해 448로 학습합니다.
- **실행 방식**: 학습은 백그라운드 프로세스(`start_trainer`, 로그 `train.log`)로 돕니다. 셀 출력이 끊겨도 학습은 계속되고, 진행 상황 셀은 언제든 실행할 수 있습니다.
  epoch마다 `last.pt`를 저장해서 이어서 학습할 수 있습니다.
- **결과 백업**: 6번 셀이 결과를 base64 zip으로 노트북 출력에 남깁니다. 노트북을 저장한 뒤
  `python extract_notebook_backup.py notebooks/BiomedCLIP_METER_Approximation.ipynb colab_backup/biomedclip`로 복원합니다.

로컬 CPU 확인: 전체 데이터 로드(RAD 3064/451, SLAKE 4919/1053/1061), 기울기 흐름, 부분집합 1 epoch 학습·평가(224와 448), 재개.
**GPU 전체 학습 결과는 미확인입니다.**

추가 파일:
- `biomedclip_meter.py`: 모델·학습 코드. 데이터 고정·검증은 `qcr_port.py`를 재사용합니다.
- `build_biomedclip_notebook.py`: 노트북 재생성
- `tests/test_biomedclip_meter.py`: 빠른 검사
- `tests/biomedclip_cpu_smoke.py`: CPU 학습 스모크 테스트
