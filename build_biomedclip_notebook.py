"""Build notebooks/BiomedCLIP_METER_Approximation.ipynb (self-contained; embeds qcr_port.py + biomedclip_meter.py)."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
cells = []


def add(kind, source):
    item = {'cell_type': kind, 'metadata': {}, 'id': f'bmc-{len(cells):02d}', 'source': source.strip() + '\n'}
    if kind == 'code':
        item.update(execution_count=None, outputs=[])
    cells.append(item)


add('markdown', '''
# BiomedCLIP + METER 근사 실험 (Expert-CFG Table 1의 BiomedCLIP 행)

> **동일 재현이 아닙니다.** BiomedCLIP 논문(Table 11)은 공개되지 않은 **ViT-B/16-448** 모델을 METER로 fine-tuning했습니다.
> 448 가중치·VQA 코드·학습 설정은 공개되지 않았습니다. 이 노트북은 **공개된 것만으로 가능한 한 가깝게** 만든 실험이며,
> 모든 결과에 `deviations`가 함께 기록됩니다.

| Dataset | 논문 Open / Closed / Overall |
|---|---|
| VQA-RAD (train 3064 / test 451) | 67.6 / 79.8 / 75.2 |
| SLAKE 영어 (train 4919 / val 1053 / test 1061) | 82.5 / 89.7 / 85.4 |

**무엇이 같고 무엇이 다른가**

| 항목 | 이 노트북 | 근거 |
|---|---|---|
| 인코더 | BiomedCLIP ViT-B/16 + PubMedBERT, 공개 **224** 가중치(HF revision 고정, 전체 fine-tuning) | 논문은 448 변형 사용(비공개) |
| Fusion | METER co-attention 6층(self→cross→FFN), CLS pooler, VQA head(1536→1536→LN→GELU→답) 직접 구현, 랜덤 초기화 | METER 공식 코드 구조 |
| 학습 설정 | AdamW lr 5e-6, cross-modal ×5, head ×50, wd 0.01, warmup 10% + linear decay, BCE×라벨 수, 질문 32 token, RAD batch 64·50 epoch, SLAKE batch 32·15 epoch, 증강 없음 | 논문에 없음 → METER 기반 의료 VQA인 **M3AE** 공식 config |
| 답 사전 | 모든 split의 정규화된 답(RAD 498개, SLAKE 222개 = M3AE와 동일) | M3AE / QCR 방식 |
| 데이터 | PubMedCLIP QCR 재현과 **같은 파일**(VQA-RAD 3064/451, SLAKE 영어) | 같은 비교 조건 |
| 평가 | 매 epoch test, **최고 epoch**(QCR·M3AE 방식) + 마지막 epoch + SLAKE validation 선택 epoch | 논문에 없음 |

`IMAGE_SIZE=448`을 고르면 METER처럼 224 위치 임베딩을 보간해 448로 fine-tuning합니다.
논문 입력 해상도에는 더 가깝지만, 448로 사전학습된 가중치와 같아지는 것은 아닙니다. 시간은 약 4배 걸립니다.

Open/Closed는 정답 `answer_type`으로 나눕니다(VQA-RAD test 179/272). PubMedCLIP QCR 코드는 질문 유형 분류기의 **예측**으로 나누므로(로그 180/271),
두 모델을 비교할 때 분모가 같은 지표는 Overall뿐입니다.

GPU 런타임에서 위부터 실행하세요. 학습은 **백그라운드 프로세스**로 돌기 때문에 셀 출력이 끊기거나 셀을 중단해도 학습은 계속됩니다.
진행 상황은 4-2 셀로 언제든 확인할 수 있습니다. epoch마다 `last.pt`를 저장하므로, 프로세스가 멈췄다면 같은 런타임에서 4-1 셀을 다시 실행해 이어서 학습합니다.
''')
add('markdown', '## 1. 환경 설치')
add('code', '''
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'timm==1.0.15'])
import torch
assert torch.cuda.is_available(), '런타임 > 런타임 유형 변경 > GPU'
print(torch.__version__, torch.cuda.get_device_name(0))
''')
add('markdown', '## 2. 코드\n\n저장소의 `qcr_port.py`(데이터 고정·검증)와 `biomedclip_meter.py`(모델·학습)가 그대로 들어 있습니다.')
code = "from pathlib import Path\nimport sys, importlib\nCODE_ROOT = Path('/content/biomedclip_code')\nCODE_ROOT.mkdir(exist_ok=True)\n"
for name in ('qcr_port.py', 'biomedclip_meter.py'):
    code += f"(CODE_ROOT / {name!r}).write_text(" + repr((root / name).read_text(encoding='utf-8')) + ", encoding='utf-8')\n"
code += ("if str(CODE_ROOT) not in sys.path: sys.path.insert(0, str(CODE_ROOT))\n"
         "import qcr_port, biomedclip_meter\nimportlib.reload(qcr_port); importlib.reload(biomedclip_meter)\n"
         "print(biomedclip_meter.BIOMEDCLIP[:2])")
add('code', code)
add('markdown', '''
## 3. 설정

- `DATASETS`: `'rad'`, `'slake'`
- `NUM_RUNS`: seed를 바꿔 반복합니다(seed = 0 + run 번호).
- `IMAGE_SIZE`: 224(기본, 공개 가중치 해상도) 또는 448(METER 방식 보간)
- `QUICK_CHECK_EPOCHS`: 동작 확인용으로 epoch를 줄입니다. 기본값과 다른 설정을 하나라도 쓰면 결과에 `full_protocol=False`로 표시됩니다.
- T4에서 메모리가 부족하면 `MICRO_BATCH`를 줄이세요. 유효 batch는 gradient accumulation으로 유지되며, 이 값은 프로토콜 변경으로 치지 않습니다.
''')
add('code', '''
import json
DATASETS = ['rad', 'slake']
NUM_RUNS = 1
IMAGE_SIZE = 224
QUICK_CHECK_EPOCHS = None
MICRO_BATCH = 16 if IMAGE_SIZE == 224 else 8
WORK = Path('/content/biomedclip_meter')
CACHE = Path('/content/hf_cache')
WORK.mkdir(parents=True, exist_ok=True)
assert set(DATASETS) <= {'rad', 'slake'} and IMAGE_SIZE in (224, 448)

def overrides_for():
    o = {}
    if IMAGE_SIZE != 224: o['image_size'] = IMAGE_SIZE
    if QUICK_CHECK_EPOCHS: o['epochs'] = QUICK_CHECK_EPOCHS
    return o
''')
add('markdown', '''
## 4. 학습 + 평가 (백그라운드)

### 4-1. 학습 시작

`DATASETS × NUM_RUNS` 작업을 차례로 실행하는 프로세스를 띄웁니다. 로그는 `train.log`, 끝난 작업은 `results.json`에 쌓입니다.
이미 실행 중이면 새로 띄우지 않고, 끝난 작업은 건너뜁니다.
중간에 멈추려면 `biomedclip_meter.stop_trainer(WORK)`를 실행하세요. 다시 시작하면 마지막으로 끝난 epoch 다음부터 이어집니다.
''')
add('code', '''
JOBS = [{'dataset': d, 'run': i, 'overrides': overrides_for()} for d in DATASETS for i in range(NUM_RUNS)]
biomedclip_meter.start_trainer(WORK, CACHE, JOBS, micro_batch=MICRO_BATCH)
''')
add('markdown', '### 4-2. 진행 상황 (학습 중 언제든 실행 가능)')
add('code', '''
import subprocess
import pandas as pd
from IPython.display import display
JOBS = json.loads((WORK / 'jobs.json').read_text(encoding='utf-8'))['jobs']
status = biomedclip_meter.trainer_status(WORK)
print('trainer:', status['state'])
display(pd.DataFrame(biomedclip_meter.progress(WORK, JOBS)))
print(subprocess.run('nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader', shell=True,
                     capture_output=True, text=True).stdout)
print(status['tail'])
''')
add('markdown', '''
### 4-3. 끝날 때까지 대기

10분마다 진행 상황을 한 줄씩 출력합니다. 이 셀을 중단해도 학습은 계속됩니다.
프로세스가 오류로 끝나면 로그 끝부분과 함께 오류를 냅니다(예: GPU 메모리 부족 → `MICRO_BATCH`를 줄이고 4-1 다시 실행).
''')
add('code', '''
import time
while biomedclip_meter.trainer_pid(WORK):
    line = '  '.join(f"{p['dataset']}/run{p['run']} {p['epochs']} best {p['best_overall']}" for p in biomedclip_meter.progress(WORK, JOBS))
    print(time.strftime('%H:%M:%S'), line, flush=True)
    time.sleep(600)
status = biomedclip_meter.trainer_status(WORK, tail_lines=40)
if status['state'] != 'finished':
    raise RuntimeError('trainer ended without finishing all jobs:\\n' + status['tail'])
print('all jobs finished')
''')
add('markdown', '## 5. 논문과 비교')
add('code', '''
from IPython.display import Markdown, display
results = json.loads((WORK / 'results.json').read_text(encoding='utf-8'))
summary = biomedclip_meter.summarize(results)
(WORK / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
names = {'best_test_epoch': 'best test epoch', 'last_epoch': 'last epoch', 'validation_selected': 'val-selected'}
lines = ['|Dataset|Runs|Protocol|Selection|Open (paper)|Closed (paper)|Overall (paper)|', '|---|---:|---|---|---:|---:|---:|']
for dataset, s in summary.items():
    p = s['paper_open_closed_overall']
    for key, label in names.items():
        if key in s:
            v = s[key]
            lines.append(f"|{dataset}|{s['runs']}|{'default' if s['full_protocol'] else 'MODIFIED'}|{label}|"
                         f"{v['open']:.2f} ({p[0]:.1f})|{v['closed']:.2f} ({p[1]:.1f})|{v['overall']:.2f} ({p[2]:.1f})|")
display(Markdown('\\n'.join(lines)))
print('근사 실험: 공개 224 가중치 + 재구현 METER + M3AE 설정. 논문 수치와의 차이는 deviations 참고.')
''')
add('markdown', '''
## 6. 결과 백업 (노트북 출력에 저장)

`results.json`, epoch별 `history.json`, `result.json`, 최고·마지막·validation 선택 epoch의 예측, 로그 끝부분을 zip으로 묶어 base64로 출력합니다.
**셀이 끝나면 노트북을 저장(Ctrl+S)**한 뒤 로컬에서 다음을 실행하면 파일이 복원됩니다. 런타임이 사라져도 결과가 남습니다.

```
python extract_notebook_backup.py notebooks/BiomedCLIP_METER_Approximation.ipynb colab_backup/biomedclip
```
''')
add('code', '''
backup = biomedclip_meter.backup_text(WORK)
print(f'{len(backup) / 1e6:.2f} MB of text')
print(backup)
''')
add('markdown', '## 7. 결과 다운로드 (Colab 브라우저에서만)\n\n기록·예측만 묶습니다. 약 2.5GB인 `last.pt`가 필요하면 `INCLUDE_CHECKPOINTS=True`로 바꾸세요.')
add('code', '''
import zipfile
from google.colab import files
INCLUDE_CHECKPOINTS = False
archive = Path('/content/biomedclip_meter_results.zip')
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for name in ('results.json', 'summary.json', 'jobs.json', 'train.log'):
        if (WORK / name).exists(): z.write(WORK / name, name)
    for path in (WORK / 'runs').rglob('*'):
        if path.is_file() and (INCLUDE_CHECKPOINTS or path.suffix not in ('.pt', '.tmp')):
            z.write(path, path.relative_to(WORK))
files.download(str(archive))
''')

notebook = {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'name': 'BiomedCLIP_METER_Approximation.ipynb', 'gpuType': 'T4', 'provenance': []},
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 5}
out = root / 'notebooks' / 'BiomedCLIP_METER_Approximation.ipynb'
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
print(out)
