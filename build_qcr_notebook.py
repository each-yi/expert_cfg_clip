"""Build notebooks/PubMedCLIP_QCR_Reproduction.ipynb (self-contained; embeds qcr_port.py)."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
cells = []


def add(kind, source):
    item = {'cell_type': kind, 'metadata': {}, 'id': f'qcr-{len(cells):02d}', 'source': source.strip() + '\n'}
    if kind == 'code':
        item.update(execution_count=None, outputs=[])
    cells.append(item)


add('markdown', '''
# PubMedCLIP + QCR 재현 (Expert-CFG Table 1의 PubMedCLIP 행)

| Dataset | 시각 인코더 | 논문 Open / Closed / Overall |
|---|---|---|
| VQA-RAD (train 3064 / test 451) | PubMedCLIP RN50 + AE | 60.1 / 80.0 / 72.1 |
| SLAKE 영어 (train 4919 / test 1061) | PubMedCLIP ViT-B/32 + AE | 78.4 / 82.5 / 80.1 |

- **학습 코드**: 공식 [sarahESL/PubMedCLIP `QCR_PubMedCLIP`](https://github.com/sarahESL/PubMedCLIP/tree/bdecf3c5a21b68200a02dc05455b3882a23206a2/QCR_PubMedCLIP)
  (commit `bdecf3c`)을 clone하고 **저자의 config 그대로** `main/main.py`를 실행합니다.
  GloVe+GRU, BAN(glimpse 10), TCR 유형 분류기, AE(재구성 손실 0.001), Adamax lr 1e-3, batch 16, 200 epoch, seed 88.
- 바뀌는 것은 최신 PyTorch/pandas/Python에서 돌리기 위한 **호환성 패치뿐**입니다. 목록과 이유는 3번 셀에서 출력되고 `run_info.json`에도 저장됩니다.
- **가중치**: 저자 HF [`sarahESL/PubMedCLIP`](https://huggingface.co/sarahESL/PubMedCLIP) (`PubMedCLIP_RN50.pth`, `PubMedCLIP_ViT32.pth`).
- **VQA-RAD 3064/451**: 원래 QCR 저자 저장소 [Awenbocc/med-vqa](https://github.com/Awenbocc/med-vqa) `data/`.
  PubMedCLIP QCR 코드는 이 묶음의 라벨 공간(487 = closed 56 + open 431)과 사전(1177단어, 동봉된 type classifier와 일치)을 전제로 합니다.
  train/test JSON·이미지 색인·AE는 HF [`aiozai/MEVF`](https://huggingface.co/datasets/aiozai/MEVF)와 바이트 단위로 같습니다.
  빠진 `images250x250.pkl`만 PubMedCLIP 스크립트로 만듭니다.
- **SLAKE**: 저자 HF [`BoKelvin/SLAKE`](https://huggingface.co/datasets/BoKelvin/SLAKE)에 PubMedCLIP `lib/utils/run.sh` 전처리를 그대로 적용합니다.
  GloVe와 AE는 HF `aiozai/MEVF`에서 받습니다.
- 모든 외부 파일은 revision/commit으로 고정하고 SHA-256을 검사합니다. Google Drive는 쓰지 않습니다.

**평가 방식(저자 코드 그대로)**: 매 epoch마다 **test set**으로 평가하고, 가장 높은 overall epoch의 점수를 보고합니다.
즉 test로 epoch를 고르는 방식이며, 이 노트북은 재현을 위해 이를 바꾸지 않습니다. 마지막 epoch 점수도 함께 기록합니다.
config 이름의 `nondeterministic`처럼 cuDNN 때문에 같은 seed로도 실행마다 결과가 조금 다릅니다. 여러 번 돌리려면 `NUM_RUNS`를 늘리세요.

GPU 런타임에서 위부터 실행하세요. 실행마다 200 epoch가 걸립니다(T4 기준 수 시간). 원 코드에는 이어서 학습하는 기능이 없습니다.
''')
add('markdown', '## 1. 환경 설치\n\nColab 기본 torch/torchvision/pandas/transformers를 그대로 씁니다. OpenAI `clip` 패키지는 commit으로 고정합니다.')
add('code', '''
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'yacs==0.1.8', 'ftfy', 'regex',
    'setuptools<81',  # clip imports pkg_resources
    'git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6'])
import torch
assert torch.cuda.is_available(), '런타임 > 런타임 유형 변경 > GPU'
print(torch.__version__, torch.cuda.get_device_name(0))
''')
add('markdown', '## 2. 포팅 코드\n\n저장소의 `qcr_port.py`가 그대로 들어 있습니다.')
source = (root / 'qcr_port.py').read_text(encoding='utf-8')
add('code', "from pathlib import Path\nimport sys\nCODE_ROOT = Path('/content/qcr_port_code')\nCODE_ROOT.mkdir(exist_ok=True)\n"
    "(CODE_ROOT / 'qcr_port.py').write_text(" + repr(source) + ", encoding='utf-8')\n"
    "if str(CODE_ROOT) not in sys.path: sys.path.insert(0, str(CODE_ROOT))\n"
    "import importlib, qcr_port\nimportlib.reload(qcr_port)\nprint('QCR commit', qcr_port.QCR_COMMIT)")
add('markdown', '''
## 3. 설정 · 공식 코드 준비

- `DATASETS`: `'rad'`(RN50), `'slake'`(ViT-B/32)
- `NUM_RUNS`: 같은 config를 반복 실행할 횟수. 결과는 실행별 최고 epoch 점수의 평균±표준편차로 요약합니다.
- `QUICK_CHECK_EPOCHS`: `None`이면 저자 설정(200 epoch)입니다. 숫자를 넣으면 동작 확인용으로 짧게 돌리며, 이 결과는 재현으로 표시되지 않습니다(`full_protocol=False`).
''')
add('code', '''
import json
DATASETS = ['rad', 'slake']
NUM_RUNS = 1
QUICK_CHECK_EPOCHS = None
WORK = Path('/content/qcr_repro')
CACHE = Path('/content/hf_cache')
WORK.mkdir(parents=True, exist_ok=True)
assert set(DATASETS) <= set(qcr_port.RUNS) and DATASETS and NUM_RUNS >= 1

QCR, PATCHES_APPLIED = qcr_port.setup_code(WORK)
for p in PATCHES_APPLIED:
    print(f"- {p['file']}: {p['reason']}")
''')
add('markdown', '## 4. 데이터 · PubMedCLIP 가중치')
add('code', '''
DATA_INFO = {d: qcr_port.prepare_data(QCR, d, CACHE) for d in DATASETS}
CLIP_WEIGHTS = {d: qcr_port.fetch(qcr_port.RUNS[d]['encoder'], CACHE) for d in DATASETS}
print(json.dumps(DATA_INFO, indent=2, ensure_ascii=False))
print({d: str(p) for d, p in CLIP_WEIGHTS.items()})
(WORK / 'run_info.json').write_text(json.dumps({
    'qcr_repo': qcr_port.QCR_REPO, 'qcr_commit': qcr_port.QCR_COMMIT, 'patches': PATCHES_APPLIED,
    'data': DATA_INFO, 'clip_weights': {d: list(qcr_port.ASSETS[qcr_port.RUNS[d]['encoder']][:4]) for d in DATASETS},
    'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
    'quick_check_epochs': QUICK_CHECK_EPOCHS, 'num_runs': NUM_RUNS}, indent=2), encoding='utf-8')
''')
add('markdown', '''
## 5. 학습 + 평가 (저자 `main/main.py`)

로그는 `output/.../medVQA.log`에 쌓이고, 결과는 `results.json`에 실행이 끝날 때마다 저장됩니다.
''')
add('code', '''
results_path = WORK / 'results.json'
results = json.loads(results_path.read_text(encoding='utf-8')) if results_path.exists() else []
done = {(r['dataset'], r['run'], r['config']) for r in results}
for dataset in DATASETS:
    for run_index in range(NUM_RUNS):
        config_name = qcr_port.make_config(QCR, dataset, CLIP_WEIGHTS[dataset], run_index, QUICK_CHECK_EPOCHS).name
        if (dataset, run_index, config_name) in done:
            print('skip finished', config_name); continue
        print('\\n===', dataset, qcr_port.RUNS[dataset]['encoder'], 'run', run_index, '===')
        result = qcr_port.train(QCR, dataset, CLIP_WEIGHTS[dataset], run_index, n_epoch=QUICK_CHECK_EPOCHS)
        results.append(result)
        results_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(json.dumps(result['best']), 'full_protocol =', result['full_protocol'])
''')
add('markdown', '## 6. 논문과 비교')
add('code', '''
from IPython.display import Markdown, display
summary = qcr_port.summarize(results)
(WORK / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
lines = ['|Dataset|Encoder|Runs|Protocol|Metric|Paper|Ours (best epoch)|Delta|Ours (last epoch)|',
         '|---|---|---:|---|---|---:|---:|---:|---:|']
for dataset, s in summary.items():
    for i, key in enumerate(('open', 'closed', 'overall')):
        paper = s['paper_open_closed_overall'][i]
        lines.append(f"|{dataset}|{qcr_port.RUNS[dataset]['encoder']}|{s['runs']}|"
                     f"{'full' if s['full_protocol'] else 'QUICK CHECK'}|{key}|{paper:.1f}|"
                     f"{s[key]:.2f} ± {s[key + '_std']:.2f}|{s[key] - paper:+.2f}|{s['last_epoch_' + key]:.2f}|")
display(Markdown('\\n'.join(lines)))
''')
add('markdown', '## 7. 결과 다운로드\n\n로그·config·결과 JSON만 묶습니다. 1GB 정도인 `*_best.pth`가 필요하면 `INCLUDE_CHECKPOINTS=True`로 바꾸세요.')
add('code', '''
import zipfile
from google.colab import files
INCLUDE_CHECKPOINTS = False
archive = Path('/content/qcr_reproduction_results.zip')
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for name in ('run_info.json', 'results.json', 'summary.json'):
        if (WORK / name).exists():
            z.write(WORK / name, name)
    for path in list((QCR / 'output').rglob('*')) + list((QCR / 'configs').glob('repro_*.yaml')):
        if path.is_file() and (INCLUDE_CHECKPOINTS or path.suffix != '.pth'):
            z.write(path, Path('qcr') / path.relative_to(QCR))
files.download(str(archive))
''')

notebook = {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'name': 'PubMedCLIP_QCR_Reproduction.ipynb', 'gpuType': 'T4', 'provenance': []},
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 5}
out = root / 'notebooks' / 'PubMedCLIP_QCR_Reproduction.ipynb'
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
print(out)
