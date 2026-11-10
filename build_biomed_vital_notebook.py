"""Build notebooks/BioMedVITAL_QLoRA_Approximation.ipynb (self-contained; embeds qcr_port.py, biomedclip_meter.py, biomed_vital_qlora.py)."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
cells = []


def add(kind, source):
    item = {'cell_type': kind, 'metadata': {}, 'id': f'bv-{len(cells):02d}', 'source': source.strip() + '\n'}
    if kind == 'code':
        item.update(execution_count=None, outputs=[])
    cells.append(item)


add('markdown', '''
# BioMed-VITAL (13B, 60K) QLoRA 근사 재현 (Expert-CFG Table 1의 BioMed-VITAL 행)

> **동일 재현이 아닙니다.** 공개된 것은 instruction-tuned 가중치 하나(HF `mao1207/BioMed-VITAL-models` @ `c93f260`)와
> 저자 추론 코드(`eval/model_vqa_med.py`)뿐입니다. VQA-RAD·SLAKE로 fine-tuning한 가중치와 fine-tuning 코드는 공개되지 않았습니다.
> 이 노트북은 그 가중치에서 **QLoRA로 downstream fine-tuning**을 하는 근사 실험입니다.

| Dataset | split (train / test) | 논문 Open / Closed / Overall |
|---|---|---|
| VQA-RAD | **원본 1,797 / 451** (MEVF `frame` 질문 제외, 논문 Table 3과 같음) | 64.88 / 84.55 / 76.74 |
| SLAKE 영어 | 4,919 / 1,061 | 87.82 / 86.54 / 87.32 |

**실행 순서 (자동)**

1. (선택, 기본 끔) fine-tuning 전 VQA-RAD test 평가 (zero-shot 기준점)
2. VQA-RAD: LoRA lr **2e-5 / 1e-4 / 2e-4** 각각 **3 epoch cosine 스케줄로 시작해 1 epoch**만 학습 → validation 평가
3. validation이 가장 좋은 lr을 골라 **저장된 상태 그대로 epoch 3까지 이어서** 학습 (처음부터 3 epoch 돌린 것과 같음) → test 평가
4. SLAKE: 고른 lr로 3 epoch → test 평가

**무엇이 같고 무엇이 다른가**

| 항목 | 이 노트북 | 근거 |
|---|---|---|
| 시작 가중치 | BioMed-VITAL instruction-tuned 13B (LLaMA-13B + Linear projector + CLIP ViT-L/14-336) | HF revision 고정 |
| 학습 방식 | **QLoRA**: 4bit NF4 + double quant, bf16 계산, LoRA r 128 / alpha 256 / dropout 0.05 (LLM의 모든 linear), projector 학습(lr 2e-5), CLIP 고정 | 논문은 전체 fine-tuning → LLaVA v1.5 `finetune_task_lora.sh` |
| epoch / batch / 스케줄 | **3 epoch**, 전역 batch 64, cosine, warmup 0.03, wd 0, grad clip 1.0 | 논문 15 epoch → LLaVA-Med v1.0.0 fine-tuning 명령 값 |
| lr | LoRA lr은 3개 중 validation으로 선택, projector 2e-5 | 교수님 지시 / 논문 2e-5는 전체 fine-tuning용 |
| 학습 텍스트·라벨 마스킹 | LLaVA-Med v1.0.0 `train.py` 그대로 (`### Human: 질문\\n<image>` / `### Assistant: 답`) | 저자 fine-tuning 데이터 비공개 |
| 이미지 토큰 | `<im_start>` + `<im_patch>`×576 + `<im_end>`, CLIP 뒤에서 두 번째 층, CLS 제외 | LLaVA-Med v1 `LlavaLlamaModel` |
| 추론 프롬프트·디코딩 | 저자 `model_vqa_med.py` (conv `default`, 샘플링 T=0.7, 최대 1024 토큰, `###`에서 중단) | 저자 코드 |
| validation | VQA-RAD train 이미지의 10% (seed 0, 이미지 단위) → train 1,626 / val 171 | 논문에 없음, test는 마지막에만 사용 |
| 채점 | **논문 방식**(LLaVA-Med `run_eval.py`: closed yes/no hit, open token recall) + **통일 방식**(정규화 EM, token F1) | 다른 모델과 비교용 |

Open/Closed는 모든 채점에서 정답 `answer_type` 기준입니다. Overall은 질문 전체 평균입니다(논문 표와 같은 가중).

**필요 자원 (추정)**: A100(40GB 이상, bf16). 모델 다운로드 약 52GB + 디스크 여유 90GB 이상.
전체 약 4~5.5시간(약 50~65 컴퓨팅 단위). 실제 속도는 학습 로그의 `s/step`과 `eta`로 확인하세요.

학습은 **백그라운드 프로세스**로 돕니다. 셀을 중단하거나 출력이 끊겨도 계속됩니다.
optimizer step 10번마다와 epoch마다 상태를 저장하므로, 프로세스가 죽으면 **같은 런타임에서** 5-1 셀을 다시 실행해 이어서 학습합니다.

**저장 (체크포인트·모델)**: VS Code에서 Colab 커널을 쓰면 파일은 PC가 아니라 Colab 런타임 디스크(`/content`)에 생기고,
런타임이 끝나면 사라집니다. 그래서 epoch마다 아래 파일을 **본인 HF 계정의 비공개 model repo**에 자동으로 올리고,
PC에서는 8번 셀의 명령으로 받습니다.

| 파일 | 내용 | 크기 |
|---|---|---|
| `runs/<run>/epoch<k>/` | fine-tuned 모델: LoRA adapter(`adapter_model.safetensors`, peft 형식) + `mm_projector.safetensors` + `meta.json` | 약 2GB |
| `runs/<run>/state.pt` | 이어서 학습용 체크포인트(LoRA·projector·optimizer·scheduler·난수 상태) | 약 6GB |
| `results/*.json`, `summary.json`, `train.log` | 결과(질문별 예측 포함)와 로그 | 작음 |

새 런타임에서 같은 `HUB_REPO`로 5-1을 실행하면 Hub에서 체크포인트와 결과를 먼저 받아 **끝난 단계는 건너뛰고 이어서** 학습합니다.
''')
add('markdown', '## 1. 환경 설치\n\n버전을 고정합니다. 설치 후 런타임을 다시 시작하라는 메시지가 나오면 다시 시작한 뒤 2번 셀부터 실행하세요.')
add('code', '''
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.46.3', 'peft==0.13.2',
                       'accelerate==1.1.1', 'bitsandbytes', 'sentencepiece'])
import shutil, torch, transformers, peft, bitsandbytes
assert torch.cuda.is_available(), '런타임 > 런타임 유형 변경 > A100 GPU'
assert torch.cuda.is_bf16_supported(), 'bf16이 되는 GPU(A100/L4/H100)가 필요합니다'
free_gb = shutil.disk_usage('/content').free / 1e9
print(torch.__version__, transformers.__version__, peft.__version__, bitsandbytes.__version__)
print(torch.cuda.get_device_name(0), f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB', f'disk free {free_gb:.0f} GB')
assert free_gb > 90, '디스크 여유가 90GB 이상 필요합니다'
''')
add('markdown', '## 2. 코드\n\n저장소의 `qcr_port.py`(데이터 고정·검증), `biomedclip_meter.py`(정규화 함수), `biomed_vital_qlora.py`(모델·학습·평가)가 그대로 들어 있습니다.')
code = "from pathlib import Path\nimport sys, importlib\nCODE_ROOT = Path('/content/biomed_vital_code')\nCODE_ROOT.mkdir(exist_ok=True)\n"
for name in ('qcr_port.py', 'biomedclip_meter.py', 'biomed_vital_qlora.py'):
    code += f"(CODE_ROOT / {name!r}).write_text(" + repr((root / name).read_text(encoding='utf-8')) + ", encoding='utf-8')\n"
code += ("if str(CODE_ROOT) not in sys.path: sys.path.insert(0, str(CODE_ROOT))\n"
         "import qcr_port, biomedclip_meter, biomed_vital_qlora as bv\n"
         "importlib.reload(qcr_port); importlib.reload(biomedclip_meter); importlib.reload(bv)\n"
         "print(bv.MODEL, bv.VISION)")
add('code', code)
add('markdown', '''
## 3. 설정

- `DATASETS`: `['rad', 'slake']`. SLAKE만 따로 돌리려면 `['slake']`와 `CHOSEN_LR`을 함께 지정하세요.
- `LRS`: 비교할 LoRA lr (교수님 지시: 2e-5, 1e-4, 2e-4)
- `CHOSEN_LR`: `None`이면 validation으로 자동 선택합니다(통일 overall EM → open F1 → 작은 lr 순). 숫자를 넣으면 그 lr을 씁니다.
- `ZERO_SHOT`: fine-tuning 전 test 평가를 할 데이터셋(기본 끔). 학습 전 모델은 답을 길게(최대 1024 토큰) 생성하므로
  VQA-RAD만으로 20~60분 걸릴 수 있습니다. 기준점이 필요하면 `['rad']`.
- `TEST_GREEDY`: test를 저자 방식(샘플링) 외에 greedy로도 한 번 더 평가합니다(재현성 확인용).
- `MICRO_BATCH`: GPU에 한 번에 올리는 개수. 전역 batch 64는 gradient accumulation으로 유지됩니다. 메모리가 부족하면 2로 줄이세요.
- `QUICK_CHECK`: `True`면 데이터 일부(train 128개)로 전체 흐름만 확인합니다(약 20~30분, 결과는 의미 없음). 별도 폴더에 저장됩니다.
- `HUB_REPO`: 체크포인트·모델을 올릴 **비공개** HF model repo(`본인아이디/이름`). 없으면 다음 셀이 비공개로 만듭니다. `None`이면 Colab 디스크에만 저장됩니다.
- `HUB_STATE`: `True`면 `state.pt`(약 6GB)도 epoch마다 올립니다(새 런타임에서 이어서 학습하려면 필요).
  같은 경로에 덮어써도 HF는 이전 버전을 기록에 남기므로 전체 실행에서 repo가 수십 GB까지 커질 수 있습니다. 모델만 필요하면 `False`.
''')
add('code', '''
import json
DATASETS = ['rad', 'slake']
LRS = [2e-5, 1e-4, 2e-4]
CHOSEN_LR = None
ZERO_SHOT = []
TEST_GREEDY = True
MICRO_BATCH = 4
QUICK_CHECK = False
HUB_REPO = None  # 예: 'my-hf-id/biomed-vital-qlora'
HUB_STATE = True
CACHE = Path('/content/hf_cache')
WORK = Path('/content/biomed_vital_quick' if QUICK_CHECK else '/content/biomed_vital')
WORK.mkdir(parents=True, exist_ok=True)
assert set(DATASETS) <= {'rad', 'slake'} and DATASETS
SPEC = {'work_dir': str(WORK), 'cache_dir': str(CACHE), 'datasets': DATASETS, 'zero_shot': ZERO_SHOT,
        'chosen_lr': CHOSEN_LR, 'test_greedy': TEST_GREEDY, 'overrides': {'lrs': LRS, 'micro_batch': MICRO_BATCH}}
if QUICK_CHECK:
    SPEC['limit'] = {'train': 128, 'validation': 24, 'test': 24}
if HUB_REPO:  # Hub 안에서도 WORK 폴더 이름(biomed_vital / biomed_vital_quick) 아래에 저장
    SPEC.update(hub_repo=HUB_REPO, hub_prefix=WORK.name, hub_state=HUB_STATE)
else:
    print('경고: HUB_REPO가 없어서 모델·체크포인트는 Colab 디스크에만 남고 런타임이 끝나면 사라집니다.')
print(json.dumps(SPEC, indent=1))
''')
add('markdown', '''
### 3-1. Hugging Face 로그인 (HUB_REPO를 쓸 때)

**write** 권한 토큰이 필요합니다(https://huggingface.co/settings/tokens). Colab 보안 비밀(`HF_TOKEN`)이 있으면 그것을 쓰고,
없으면 입력창이 뜹니다. 토큰은 환경변수로만 학습 프로세스에 전달되고 `spec.json`이나 출력에는 남지 않습니다.
''')
add('code', '''
import os, getpass
from huggingface_hub import HfApi
if HUB_REPO:
    if not os.environ.get('HF_TOKEN'):
        try:
            from google.colab import userdata
            os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
        except Exception:
            os.environ['HF_TOKEN'] = getpass.getpass('HF write token: ').strip()
    api = HfApi()
    print('logged in as', api.whoami()['name'])
    url = api.create_repo(HUB_REPO, repo_type='model', private=True, exist_ok=True)
    info = api.model_info(HUB_REPO)
    assert info.private, f'{HUB_REPO} is PUBLIC - use a private repo'
    print('hub repo (private):', url)
else:
    print('HUB_REPO = None: Hub 저장 안 함')
''')
add('markdown', '''
## 4. 데이터 확인

VQA-RAD 원본 split과 validation 분할, SLAKE 영어 split의 개수를 확인합니다.
토크나이저만 받아서 학습 예시 하나의 **학습 대상 토큰**과 평가 프롬프트를 보여줍니다.
''')
add('code', '''
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
for d in DATASETS:
    print(d, json.dumps(bv.data_summary(bv.load_data(d, WORK, CACHE, bv.config_for(SPEC['overrides']))), indent=1))
tok_dir = Path(hf_hub_download(bv.MODEL[0], 'tokenizer.model', revision=bv.MODEL[1], cache_dir=str(CACHE))).parent
for f in ('tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json'):
    hf_hub_download(bv.MODEL[0], f, revision=bv.MODEL[1], cache_dir=str(CACHE))
tok = AutoTokenizer.from_pretrained(tok_dir, use_fast=False)
ids, labels = bv.build_train_example(tok, 'Is there a rib fracture?', 'No', 576, 2048)
print('train tokens', len(ids), '| supervised:', repr(tok.decode([t for t, l in zip(ids, labels) if l != bv.IGNORE_INDEX])))
print('eval prompt (image tokens shortened):')
print(bv.build_eval_prompt('Is there a rib fracture?', 3))
''')
add('markdown', '''
## 5. 학습 + 평가 (백그라운드)

### 5-1. 시작

처음에는 약 52GB 모델을 받고 4bit로 올리는 데 20~30분 걸립니다(로그에 진행 표시).
이미 실행 중이면 새로 띄우지 않고, 끝난 단계는 건너뜁니다. 멈추려면 `bv.stop_trainer(WORK)`.
`HUB_REPO`가 있으면 모델을 받기 전에 Hub에 있는 체크포인트·결과를 먼저 내려받고(`hub: restored ...`),
epoch이 끝날 때마다 올린 기록이 로그에 `hub: ...`로 찍힙니다(epoch당 1~3분). 업로드가 실패해도 학습은 계속되고, 8번 셀로 다시 올릴 수 있습니다.
''')
add('code', '''
bv.start_trainer(SPEC)
''')
add('markdown', '### 5-2. 진행 상황 (언제든 실행 가능)\n\n학습 로그의 `s/step`과 `eta`로 남은 시간을 볼 수 있습니다. VQA-RAD는 epoch당 26 step, SLAKE는 77 step입니다.')
add('code', '''
import subprocess
import pandas as pd
from IPython.display import display
status = bv.trainer_status(WORK, tail_lines=20)
print('pipeline:', status['state'])
rows = bv.progress(WORK)
if rows: display(pd.DataFrame(rows))
print(subprocess.run('nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader', shell=True,
                     capture_output=True, text=True).stdout)
print(status['tail'])
''')
add('markdown', '### 5-3. 끝날 때까지 대기\n\n10분마다 로그 마지막 줄을 출력합니다. 이 셀을 중단해도 학습은 계속됩니다.')
add('code', '''
import time
while bv.trainer_pid(WORK):
    tail = bv.trainer_status(WORK, tail_lines=1)['tail']
    print(time.strftime('%H:%M:%S'), tail[-200:], flush=True)
    time.sleep(600)
status = bv.trainer_status(WORK, tail_lines=40)
if status['state'] != 'finished':
    raise RuntimeError('pipeline ended without finishing:\\n' + status['tail'])
print('all stages finished')
''')
add('markdown', '''
## 6. 결과

- **논문 방식**: LLaVA-Med `run_eval.py` (closed: 답에 yes/no가 있고 정답이 포함되면 정답, open: 정답 토큰 recall). 논문 수치와 비교하는 값입니다.
- **통일 방식**: 앞의 `Assistant:` 태그만 떼고 정규화한 exact match(closed 정확도, open EM)와 token F1. 분류 모델(PubMedCLIP, BiomedCLIP)과 비교하는 값입니다.
''')
add('code', '''
from IPython.display import Markdown, display
summary = bv.summarize(WORK)
(WORK / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
if summary['lr_selection']:
    print('chosen lr:', summary['lr_selection']['chosen_lr'], '| rule:', summary['lr_selection']['rule'])
lines = ['|Result|Split|Decoding|Open (paper)|Closed (paper)|Overall (paper)|Closed acc|Open EM|Open F1|Overall EM|',
         '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
for name, m in summary['results'].items():
    p, u = m['paper'], m['unified']
    lines.append(f"|{name}|{m['split']}|{m['decoding']}|{p['open']:.2f}|{p['closed']:.2f}|{p['overall']:.2f}|"
                 f"{u['closed_acc']:.2f}|{u['open_em']:.2f}|{u['open_f1']:.2f}|{u['overall_em']:.2f}|")
for d, (o, c, a) in bv.PAPER.items():
    lines.append(f'|**paper** {d}|test|sampling|{o:.2f}|{c:.2f}|{a:.2f}|||||')
display(Markdown('\\n'.join(lines)))
print('근사 실험: QLoRA, 3 epoch, lr은 validation 선택. 차이는 summary.json의 deviations 참고.')
''')
add('markdown', '''
## 7. 결과 백업 (노트북 출력에 저장)

결과 JSON(질문별 예측 포함), 진행 기록, 로그 끝부분을 zip으로 묶어 base64로 출력합니다. LoRA 가중치(수 GB)는 들어가지 않습니다.
**셀이 끝나면 노트북을 저장(Ctrl+S)**한 뒤 로컬에서 다음을 실행하면 파일이 복원됩니다.

```
python extract_notebook_backup.py notebooks/BioMedVITAL_QLoRA_Approximation.ipynb colab_backup/biomed_vital
```
''')
add('code', '''
backup = bv.backup_text(WORK)
print(f'{len(backup) / 1e6:.2f} MB of text')
print(backup)
''')
add('markdown', '''
## 8. 체크포인트·모델을 PC로 받기 (Hugging Face Hub)

### 8-1. Hub에 빠진 파일 다시 올리기 (Colab에서)

학습 중 업로드가 실패했거나 마지막 로그까지 올리고 싶을 때 실행합니다. 이미 같은 내용이 있는 파일은 다시 보내지 않습니다.
''')
add('code', '''
assert HUB_REPO, '3번 셀에서 HUB_REPO를 지정하세요'
local = bv.hub_files(WORK, HUB_STATE)
print(len(local), 'files on the Colab disk')
assert bv.hub_sync(SPEC), 'upload failed - 로그를 보고 다시 실행하세요'
remote = [f for f in HfApi().list_repo_files(HUB_REPO) if f.startswith(WORK.name + '/')]
missing = sorted(set(local) - {f[len(WORK.name) + 1:] for f in remote})
print(len(remote), 'files on the Hub under', WORK.name + '/', '| missing:', missing or 'none')
for f in sorted(remote):
    if f.endswith(('adapter_model.safetensors', 'state.pt')):
        print(' ', f)
''')
add('markdown', '''
### 8-2. PC에서 받기 (VS Code 터미널, Colab 커널이 아니라 로컬에서 실행)

```
pip install -U huggingface_hub
hf auth login
hf download <HUB_REPO> --local-dir colab_backup/biomed_vital_hub --exclude "*state.pt"
```

- 모델만 받습니다(run·epoch마다 약 2GB). 이어서 학습할 체크포인트(`state.pt`, 약 6GB)까지 받으려면 `--exclude` 부분을 지우세요.
- 특정 run만: `--include "biomed_vital/runs/rad_lr2e-4/epoch3/*" "biomed_vital/results/*"`
- 받은 모델은 Colab에서 모델을 올린 뒤(`c = bv.build_components(...)`) `bv.load_model(c, '<폴더>/epoch3')`로 다시 넣어 평가할 수 있습니다.
''')

notebook = {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'name': 'BioMedVITAL_QLoRA_Approximation.ipynb', 'gpuType': 'A100', 'provenance': []},
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 5}
out = root / 'notebooks' / 'BioMedVITAL_QLoRA_Approximation.ipynb'
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
print(out)
