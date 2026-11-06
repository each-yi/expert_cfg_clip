# VQA-RAD · PubMedCLIP / BiomedCLIP

현재 실행 파일은 **[Medical_METER_224_VQARAD_SLAKE.ipynb](notebooks/Medical_METER_224_VQARAD_SLAKE.ipynb)** 입니다.

공개 224 HF 가중치에 METER-style 양방향 token co-attention을 붙이고 인코더까지 fine-tuning합니다. **VQA-RAD와 SLAKE만** 지원합니다. 학습 코드가 노트북에 포함되어 파일 하나만 Colab에 업로드하면 됩니다. GPU를 선택하고 첫 셀부터 실행하세요.

- `MODELS=['biomedclip','pubmedclip']`, `DATASETS=['vqa_rad','slake']`로 기본 네 조합 실행
- 원래 answer_type이 있는 HF 자료 사용. SLAKE는 영어만 사용
- validation으로 checkpoint 선택 후 test 평가, checkpoint/RNG/optimizer 저장
- 기본 224px, microbatch2, accumulation16, token co-attention6층, AMP 및 gradient checkpointing
- 결과 `/content/medical_meter`; 마지막 셀에서 ZIP 다운로드. 외부 저장소 마운트 없음
- `medical_meter.py`: 데이터·모델·학습 코드. 수정 후 `python build_meter_notebook.py`로 notebook 재생성
- `tests/test_medical_meter.py`: 작은 가짜 encoder를 이용한 실제 co-attention forward/backward·마스킹·데이터 처리 CPU 검사

CPU 회귀 검사 6개 통과. 전체 HF 모델의 Colab GPU 학습·성능은 미검증이며 노트북이 실행 시작 시 실제 GPU forward/backward를 검사합니다.

이는 원본448가중치/의료METER설정 대신 공개224가중치와 명시적인 adaptation 설정으로 진행하는 실험입니다. PubMedCLIP 옵션도 METER-style이며 원 논문 QCR은 아닙니다. 원 논문과의 차이는 run.json에 저장합니다. 정확한 원 논문 조사 내용은 [PAPER_REPRODUCTION.md](PAPER_REPRODUCTION.md)에 있습니다.

아래는 앞서 작성한 **이전 frozen-MLP 버전**의 설명입니다. 새 실험에는 위 METER-style 노트북을 사용하세요.

사용자가 제공한 `VQA_RAD_expert_cfg_reproduction (1).ipynb` 기반입니다.

## 실행

[실행 노트북](notebooks/VQA_RAD_PubMedCLIP_BiomedCLIP.ipynb)을 Colab에 업로드하고 GPU 런타임에서 위에서부터 실행하세요.
설정 셀의 `MODEL_NAME`을 `pubmedclip` 또는 `biomedclip`으로 선택합니다.
필요한 패키지는 노트북에서 설치하고 모델은 Hugging Face에서 불러옵니다.

데이터는 Hugging Face의 `flaviagiammarino/vqa-rad` train/test에서만 불러옵니다. Google Drive 연결은 사용하지 않습니다. 원본 채점 함수, 5·8·10 epoch checkpoint 저장, 자동 resume, 평가 JSON, 손실 곡선과 불확실성 분석을 포함합니다. 결과는 Colab 로컬 `/content/clip_vqarad`에 저장하며 마지막 셀에서 ZIP으로 다운로드합니다. 런타임 삭제 시 로컬 파일도 삭제됩니다.
인코더를 고정하고 MLP 답변 분류기를 학습하는 구성으로, 각 원 논문의 전용 VQA 구조를 정확히 재현한 구현은 아닙니다. 실제 GPU 실행은 미검증입니다.

## 파일

- `notebooks/VQA_RAD_PubMedCLIP_BiomedCLIP.ipynb`: 독립 실행 노트북
- `build_vqarad_notebook.py`: 첨부 원본을 읽어 노트북을 재생성하는 스크립트. 재생성 시 원본 경로가 필요합니다.
- `diagnose_vqarad.py`: 노트북 6.1절에 포함되는 Open 0점 진단 코드. 변경 후 노트북을 재생성하세요.

## Open 0점 진단

노트북 **6.1 Open 0점 원인 분석**의 정의·실행 셀을 현재 학습이 끝난 런타임에 복사하여 실행하세요. 재학습은 필요 없습니다. 저장된 예측, 답변 vocabulary, 학습·평가 성능, 정답 클래스 순위, yes/no 확률, feature 통계와 이미지·질문 shuffle 결과를 `.diagnostics.json`에 저장합니다. Oracle type mask 결과는 진단 전용이며 benchmark 점수가 아닙니다.

비교값은 Expert-CFG 논문 Table 1의 VQA-RAD PubMedCLIP(60.10/80.00/72.10), BioMedCLIP(67.60/79.80/75.20)입니다. HF mirror의 유형 추정과 frozen MLP 구성은 원 논문과 다른 프로토콜입니다.

원본 노트북과 상위 폴더의 다른 프로젝트는 변경하지 않았습니다.
