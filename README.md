# VQA-RAD · PubMedCLIP / BiomedCLIP

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

원본 노트북과 상위 폴더의 다른 프로젝트는 변경하지 않았습니다.
