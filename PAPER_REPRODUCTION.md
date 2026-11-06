# 동일 재현 조사 결과

현재 **동일 재현 학습 구현은 완료되지 않았습니다**. 기존 frozen-MLP 노트북은 탐색용이며 원 논문 baseline이 아닙니다.
사용자가 공개224가중치 기반 adaptation 진행을 선택하여, [METER-style 실행 노트북](notebooks/Medical_METER_224_VQARAD_SLAKE.ipynb)을 추가했습니다. 이 노트북은 원본448과의 차이를 명시하고 인코더까지 fine-tuning합니다. 별도의 동일 재현 완료 주장은 하지 않습니다.

## 중요한 정정: PubMedCLIP은 dataset마다 backbone이 다름

원 논문 Table 1(p.1186)을 확인하면 VQA-RAD 60.1/80.0/72.1은 **RN50 + AE**, SLAKE
78.4/82.5/80.1은 **ViT-B/32 + AE**입니다. 하나의 HF ViT 체크포인트로 두 값을 재현할 수 없습니다.
질문 인코더는 GloVe+RNN이며 CLIP의 텍스트 인코더를 질문 인코더로 쓰는 실험과 다릅니다.

공식 QCR 코드를 실제로 내려받아 확인한 commit:
`bdecf3c5a21b68200a02dc05455b3882a23206a2`.

- GRU 1024, 질문 길이 12, TF-IDF/CAT GloVe
- Open/Closed별 BAN, glimpse 10 및 질문 유형 모델
- 사전학습 AE + 재구성 손실(가중치 0.001)
- 공식 코드: Adamax, LR 0.001, batch16, 200epochs, seed88
- 논문은 실험 10회 평균을 보고. 본문은 Adam이라고 표현하지만 공개 구현은 Adamax 사용. 이 차이도 기록해야 함.
- AE checkpoint, GloVe 초기화/TF-IDF 파일, 원래 dictionary 및 label mapping이 추가로 필요함.
  저장소의 pretrained 질문 유형 모델과 dictionary 순서를 맞춰야 하며 새 dictionary를 무조건 생성하면 안 됨.
- 공식 코드는 PyTorch1.8/torchvision0.9 등 구버전 기준이며 Colab Python3.13에 그대로 설치할 수 없음.
  현대화 port는 forward/loss/전처리 동등성 검증 후에 동일 구현으로 볼 수 있음.

## BioMedCLIP

원 논문 §3.4/Table11의 모델은 METER co-attention VQA + ViT-B/16-448-PMB/256입니다.
공식 HF `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`는 224 체크포인트이며,
일반 METER 저장소는 의료 VQA 전용 설정/학습 코드를 대체하지 않습니다.
현재 조사에서는 448 원본 checkpoint 및 의료 VQA METER 전용 학습 설정을 확보하지 못했습니다.
논문 Appendix A의 40epochs 등은 **CLIP 사전학습 설정**이므로 VQA downstream에 복사하면 안 됩니다.
입력 resize/position embedding 보간만으로 224 사전학습 가중치를 448 원본 가중치로 만들 수 없습니다.

## 데이터·평가

- 데이터는 HF에서만 내려받고 Google Drive는 사용하지 않습니다.
- 기존 flaviagiammarino/vqa-rad는 train1793/test451이고 원본 answer_type이 없어 동일 재현에 부족합니다.
- 원 논문 표기 RAD train3064/test451 및 MEVF split, 원래 answer_type을 확인해야 합니다.
- metadata 보존 HF 후보 `abhay2812/vqa-rad`도 표기상 train1793이므로 원본 유형을 복원해도 학습 split은 여전히 다릅니다.
- SLAKE는 `BoKelvin/SLAKE`의 원래 JSON/이미지를 HF에서 받으며 영어와 original answer_type을 유지해야 합니다.
- PubMedCLIP 분류 baseline은 원래 label mapping/전처리 기반 정확도가 필요합니다.
  기존 Phi용 Open token recall을 유일한 재현 지표로 사용하지 않습니다.
- Open/Closed 원본 유형을 yes/no 여부로 추정하거나 test를 train에 합쳐 개수를 맞추지 않습니다.

## 필요한 추가 자산

1. PubMedCLIP RN50 HF 원본 가중치와 QCR 전처리 bundle(AE/GloVe/dictionary/label/TF-IDF; 출처 확인).
2. MEVF의 RAD train3064/test451 split 및 image/answer_type을 보존한 HF 자료.
3. BioMedCLIP 448 원본 가중치와 해당 의료 VQA METER 코드·config.

이 자산 없이 같은 성능을 보장하거나 동일 재현 완료로 표시할 수 없습니다.

## 출처

- [Expert-CFG Table1](https://arxiv.org/html/2507.09209v1)
- [PubMedCLIP 원 논문 Table1 및 §5.1](https://aclanthology.org/2023.findings-eacl.88.pdf)
- [공식 QCR 코드](https://github.com/sarahESL/PubMedCLIP/tree/bdecf3c5a21b68200a02dc05455b3882a23206a2/QCR_PubMedCLIP)
- [BioMedCLIP §3.4/Table11](https://arxiv.org/html/2303.00915v1)
- [일반 METER](https://github.com/zdou0830/METER)
- [공식 공개 224 HF 모델](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
