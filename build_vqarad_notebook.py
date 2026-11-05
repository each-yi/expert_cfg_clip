"""Adapt the explicitly supplied VQA_RAD_expert_cfg_reproduction notebook."""
import ast
import copy
import json
from pathlib import Path

source = Path(r'C:\Users\82105\Downloads\VQA_RAD_expert_cfg_reproduction (1).ipynb')
original = json.loads(source.read_text(encoding='utf-8'))
cells = []


def add(kind, text):
    c = dict(cell_type=kind, metadata={}, source=text.strip()+'\n', id=f'vqarad-clip-{len(cells):02}')
    if kind == 'code':
        c.update(execution_count=None, outputs=[])
    cells.append(c)


def src(i):
    return ''.join(original['cells'][i]['source'])


add('markdown', '''
# VQA-RAD — PubMedCLIP / BiomedCLIP baseline

**기준 파일:** `VQA_RAD_expert_cfg_reproduction (1).ipynb`.
첨부 노트북의 normalize/closed_accuracy/open_recall/summarize 함수와
5·8·10 epoch checkpoint 비교, 자동 resume, 손실 곡선, 불확실성 분석 흐름을 유지합니다.
데이터와 모델은 Hugging Face에서만 불러오며 결과는 `/content/clip_vqarad`에 저장합니다.
Colab 런타임 삭제 시 로컬 결과도 삭제되므로 마지막 다운로드 셀로 보관하세요.

달라진 부분은 Phi/LoRA 대신 **HF CLIP 인코더 + 답변 분류기**라는 점입니다.
이미지·질문 인코더를 고정하고 MLP만 학습합니다. 원 논문의 각 CLIP VQA head를 정확히 재현한 구현은 아닙니다.
Colab GPU 런타임에서 위에서부터 실행하세요. 모델을 바꿔 실행할 때는 런타임을 새로 시작하는 것을 권장합니다.
''')
add('markdown', '## 0. 환경 준비\n\nCLIP은 Phi 호환 패치나 Expert-CFG 저장소 clone이 필요하지 않습니다.')
add('code', '''
import subprocess, sys, importlib.metadata
# 모델 로더와 Trainer를 같은 검증 대상 API 버전으로 맞춥니다.
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'transformers==4.48.3', 'open_clip_torch==2.30.0',
    'datasets>=3.2,<4', 'huggingface_hub>=0.28,<1', 'Pillow>=10.4',
    'scikit-learn', 'matplotlib', 'safetensors', 'accelerate>=0.26'])
if any(name in sys.modules for name in ('transformers', 'open_clip', 'datasets')):
    raise RuntimeError('이미 import된 패키지가 있습니다. 런타임 → 세션 다시 시작 후 위에서부터 다시 실행하세요.')
''')
add('code', '''
import os, json, glob, random, inspect, re, gc, hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from tqdm.auto import tqdm
from transformers import Trainer, TrainingArguments, TrainerCallback
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
assert DEVICE == 'cuda', '런타임 → 런타임 유형 변경 → GPU를 선택하세요.'
print('torch:', torch.__version__, '| device:', DEVICE)
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
''')
add('markdown', '## 1. 설정\n\nMODEL_NAME을 바꿔 두 모델을 각각 실행합니다. 기존 Phi 결과와 다른 파일명으로 저장합니다.')
add('code', '''
MODEL_NAME = 'biomedclip'  # 'pubmedclip' 또는 'biomedclip'
MODEL_IDS = {
    'pubmedclip': 'flaviagiammarino/pubmed-clip-vit-base-patch32',
    'biomedclip': 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
}
assert MODEL_NAME in MODEL_IDS
HF_DATASET = 'flaviagiammarino/vqa-rad'
CACHE_DIR = '/content/hf_cache'
EPOCHS = 10
SAVE_EPOCHS = {5, 8, 10}
LR = 1e-4
GLOBAL_BATCH = 64
PER_DEVICE_BS = 32
GRAD_ACCUM = GLOBAL_BATCH // PER_DEVICE_BS
ENCODE_BATCH = 32
HIDDEN = 512
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0
SEED = 42
_tag = f'{MODEL_NAME}_frozen_mlp_e{EPOCHS}'
OUTPUT_DIR = f'/content/clip_vqarad/logs_{_tag}'
RESULT_JSON = f'/content/clip_vqarad/vqarad_{_tag}_test.json'
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print('OUTPUT_DIR:', OUTPUT_DIR)
print('클립 전용 설정: frozen encoders + MLP; Phi LoRA와 다른 학습 실험입니다.')
''')
add('markdown', '''
## 2. 데이터 — Hugging Face 전용

HF의 train/test split을 그대로 사용합니다. 로컬 JSON·OSF·수동 분할 fallback은 없습니다.
미러에 answer_type이 없는 경우 yes/no를 CLOSED, 나머지를 OPEN으로 분류합니다.
원본 JSON의 answer_type 및 451/179/272 구성과 다를 수 있으므로 실제 개수를 출력합니다.
''')
add('code', '''
from datasets import load_dataset, Image as HFImage
from huggingface_hub import HfApi
import io
DATA_REVISION = HfApi().dataset_info(HF_DATASET).sha
hf_data = load_dataset(HF_DATASET, revision=DATA_REVISION, cache_dir=CACHE_DIR)
assert 'train' in hf_data and 'test' in hf_data, 'HF 데이터셋에 train/test가 필요합니다.'

class VQARADSplit(torch.utils.data.Dataset):
    def __init__(self, split):
        ds = hf_data[split]
        required = {'image', 'question', 'answer'}
        assert required.issubset(ds.column_names), f'예상하지 못한 HF schema: {ds.column_names}'
        self.records = ds.cast_column('image', HFImage(decode=False))
        assert len(self.records), f'빈 split: {split}'
    def __len__(self): return len(self.records)
    def annotation(self, idx):
        r = self.records[idx]
        q, a = str(r['question']).strip(), str(r['answer']).strip()
        assert q and a and r['question'] is not None and r['answer'] is not None, f'빈 QA: {idx}'
        at = str(r.get('answer_type', '')).upper()
        if at not in ('OPEN','CLOSED'):
            at = 'CLOSED' if a.lower() in ('yes','no') else 'OPEN'
        return {'question': q, 'answer': a, 'answer_type': at}
    def __getitem__(self, idx):
        payload = self.records[idx]['image']
        if payload is None: raise ValueError(f'누락된 HF 이미지: {idx}')
        if payload.get('bytes') is not None:
            source = io.BytesIO(payload['bytes'])
        elif payload.get('path'):
            source = payload['path']
        else:
            raise ValueError(f'잘못된 HF 이미지: {idx}')
        with Image.open(source) as im:
            image = im.convert('RGB')
        return {**self.annotation(idx), 'image': image}

def type_counts(ds):
    closed = sum(ds.annotation(i)['answer_type']=='CLOSED' for i in range(len(ds)))
    return closed, len(ds)-closed

train_ds, test_ds = VQARADSplit('train'), VQARADSplit('test')
for name, ds in [('train',train_ds),('test',test_ds)]:
    print(name, 'total:',len(ds),'closed/open:',type_counts(ds))
print('HF source:',HF_DATASET,'revision:',DATA_REVISION)
''')
add('markdown', '## 3. Hugging Face 모델 로드\n\n전용 전처리로 이미지와 질문을 각각 인코딩합니다. Phi 가중치는 다운로드하지 않습니다.')
add('code', '''
if MODEL_NAME == 'pubmedclip':
    from transformers import CLIPModel, CLIPProcessor
    encoder = CLIPModel.from_pretrained(MODEL_IDS[MODEL_NAME]).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_IDS[MODEL_NAME])
else:
    from open_clip import create_model_from_pretrained, get_tokenizer
    encoder, preprocess = create_model_from_pretrained('hf-hub:'+MODEL_IDS[MODEL_NAME])
    encoder = encoder.to(DEVICE).eval()
    tokenizer = get_tokenizer('hf-hub:'+MODEL_IDS[MODEL_NAME])
encoder.requires_grad_(False)

@torch.inference_mode()
def encode_dataset(ds):
    result = []
    for start in tqdm(range(0, len(ds), ENCODE_BATCH), desc='Encoding'):
        batch = [ds[i] for i in range(start, min(start+ENCODE_BATCH, len(ds)))]
        images = [r['image'] for r in batch]
        questions = [r['question'] for r in batch]
        if MODEL_NAME == 'pubmedclip':
            pixels = processor(images=images, return_tensors='pt').to(DEVICE)
            tokens = processor.tokenizer(questions, padding=True, truncation=True,
                max_length=77, return_tensors='pt').to(DEVICE)
            im = encoder.get_image_features(**pixels)
            tx = encoder.get_text_features(**tokens)
        else:
            pixels = torch.stack([preprocess(im) for im in images]).to(DEVICE)
            tokens = tokenizer(questions, context_length=256).to(DEVICE)
            im = encoder.encode_image(pixels)
            tx = encoder.encode_text(tokens)
        im, tx = F.normalize(im.float(), dim=-1), F.normalize(tx.float(), dim=-1)
        result.append(torch.cat([im, tx, im*tx], dim=-1).cpu())
    return torch.cat(result)

train_features = encode_dataset(train_ds).clone()
test_features = encode_dataset(test_ds).clone()
del encoder
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
print('features:', train_features.shape, test_features.shape)
''')
add('markdown', '## 4. 기존 채점 함수·분류기\n\n답변 vocabulary는 train에서만 만듭니다. Test에만 있는 답변도 평가에 포함합니다.')
tree = ast.parse(src(17))
functions = [ast.get_source_segment(src(17), node) for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in ('normalize', 'closed_accuracy', 'open_recall', 'summarize')]
add('code', '\n\n'.join(functions))
add('code', '''
def answer_key(a):
    return ' '.join(str(a).lower().strip().split())
train_examples = [train_ds.annotation(i) for i in range(len(train_ds))]
test_examples = [test_ds.annotation(i) for i in range(len(test_ds))]
answers = sorted({answer_key(r['answer']) for r in train_examples})
answer_to_id = {a: i for i, a in enumerate(answers)}

class FeatureDataset(torch.utils.data.Dataset):
    def __len__(self): return len(train_examples)
    def __getitem__(self, i):
        return {'features': train_features[i], 'labels': answer_to_id[answer_key(train_examples[i]['answer'])]}

class VQAClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(train_features.shape[1], HIDDEN), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(HIDDEN, len(answers)))
    def forward(self, features, labels=None):
        logits = self.head(features)
        result = {'logits': logits}
        if labels is not None: result['loss'] = F.cross_entropy(logits, labels)
        return result

def feature_collator(batch):
    return {'features': torch.stack([r['features'] for r in batch]),
            'labels': torch.tensor([r['labels'] for r in batch], dtype=torch.long)}

model = VQAClassifier()
# 같은 데이터·vocabulary·설정만 resume 허용
signature = {'model': MODEL_NAME, 'model_id': MODEL_IDS[MODEL_NAME], 'answers': answers,
    'dataset': HF_DATASET, 'dataset_revision': DATA_REVISION,
    'hidden': HIDDEN, 'epochs': EPOCHS, 'lr': LR, 'batch': GLOBAL_BATCH,
    'per_device_batch': PER_DEVICE_BS, 'grad_accum': GRAD_ACCUM,
    'warmup_ratio': WARMUP_RATIO, 'weight_decay': WEIGHT_DECAY,
    'seed': SEED, 'save_epochs': sorted(SAVE_EPOCHS),
    'train_features_sha256': hashlib.sha256(train_features.numpy().tobytes()).hexdigest(),
    'train_qa_sha256': hashlib.sha256(json.dumps([(r['question'],r['answer']) for r in train_examples]).encode()).hexdigest()}
config_path = Path(OUTPUT_DIR)/'clip_run.json'
if config_path.exists():
    assert json.loads(config_path.read_text()) == signature, '설정이 바뀌었습니다. _tag/OUTPUT_DIR를 변경하세요.'
else:
    assert not list(Path(OUTPUT_DIR).glob('checkpoint-*')), '기존 checkpoint와 섞지 마세요.'
    config_path.write_text(json.dumps(signature, indent=2, ensure_ascii=False), encoding='utf-8')
print('학습 답변 수:', len(answers))
''')
add('markdown', '## 5. 학습 — 선택 epoch 저장·자동 resume\n\n기존 Trainer 흐름이며 optimizer/scheduler 상태를 함께 저장합니다.')
add('code', '''
class SaveAtEpochs(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_save = int(round(state.epoch)) in SAVE_EPOCHS
        return control

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR, num_train_epochs=EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BS, gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR, optim='adamw_torch', lr_scheduler_type='cosine',
    warmup_ratio=WARMUP_RATIO, weight_decay=WEIGHT_DECAY,
    fp16=False, bf16=False, logging_steps=10,
    save_strategy='epoch', save_total_limit=None, save_only_model=False,
    remove_unused_columns=False, dataloader_num_workers=0, report_to='none', seed=SEED)
trainer = Trainer(model=model, args=training_args, train_dataset=FeatureDataset(),
    data_collator=feature_collator, callbacks=[SaveAtEpochs()])
trainer.model_accepts_loss_kwargs = False
_cks = sorted(glob.glob(os.path.join(OUTPUT_DIR, 'checkpoint-*')), key=lambda d: int(d.rsplit('-',1)[1]))
_resume = _cks[-1] if _cks else False
print('resume_from_checkpoint =', _resume)
trainer.train(resume_from_checkpoint=_resume)
trainer.save_model(OUTPUT_DIR)
trainer.save_state()
print('학습 완료')
''')
add('markdown', '''
## 6. 평가 — 원본과 같은 records JSON

idx/question/answer/answer_type/prediction/score/entropy 필드를 유지합니다.
여기서 entropy는 **답변 클래스 분포의 엔트로피**이며 Phi 생성 토큰 entropy와 다릅니다.
논문의 CLIP 보고값은 참고값이며 학습 구조가 같다는 의미는 아닙니다.
''')
add('code', '''
@torch.no_grad()
def evaluate_clip(m):
    m.to(DEVICE).eval()
    records = []
    for start in tqdm(range(0, len(test_examples), ENCODE_BATCH), desc='Evaluating'):
        logits = m(test_features[start:start+ENCODE_BATCH].to(DEVICE))['logits'].float()
        logp = logits.log_softmax(-1)
        entropy = (-(logp.exp()*logp).sum(-1)).cpu().tolist()
        ids = logits.argmax(-1).cpu().tolist()
        for offset, (label, ent) in enumerate(zip(ids, entropy)):
            i = start + offset
            ex = test_examples[i]
            pred = answers[label]
            s = (closed_accuracy if ex['answer_type']=='CLOSED' else open_recall)(pred, ex['answer'])
            records.append({'idx': i, 'question': ex['question'], 'answer': ex['answer'],
                'answer_type': ex['answer_type'], 'prediction': pred, 'score': s,
                'entropy': ent, 'entropy_type': 'answer_class_distribution', 'model': MODEL_NAME})
    return records

records = evaluate_clip(model)
Path(RESULT_JSON).write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
m = summarize(records)
paper = {'pubmedclip': {'open':60.10,'closed':80.00,'overall':72.10},
         'biomedclip': {'open':67.60,'closed':79.80,'overall':75.20}}[MODEL_NAME]
print('metric       paper      ours      diff')
for k in ('open','closed','overall'):
    print(f'{k:8s} {paper[k]:9.2f} {m[k]:9.2f} {m[k]-paper[k]:+9.2f}')
print('n_open:',m['n_open'],'n_closed:',m['n_closed'])
print('결과 저장:',RESULT_JSON)
''')
add('code', '''
# checkpoint별 평가: 저장된 답변 분류기만 교체합니다.
from safetensors.torch import load_model
_cks = sorted(glob.glob(os.path.join(OUTPUT_DIR,'checkpoint-*')), key=lambda d:int(d.rsplit('-',1)[1]))
_rows = []
for d in _cks:
    state = json.loads((Path(d)/'trainer_state.json').read_text())
    step, epoch = state['global_step'], state['epoch']
    ck_model = VQAClassifier()
    load_model(ck_model, str(Path(d)/'model.safetensors'))
    recs = evaluate_clip(ck_model)
    out = Path(RESULT_JSON).with_name(f'vqarad_{_tag}_ckpt{step}_test.json')
    out.write_text(json.dumps(recs, ensure_ascii=False), encoding='utf-8')
    _rows.append((epoch, step, summarize(recs)))
    del ck_model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
print('epoch  step      open    closed   overall')
for ep, st, m in _rows:
    print(f'{ep:5.1f} {st:5d} {m["open"]:9.2f} {m["closed"]:9.2f} {m["overall"]:9.2f}')
''')
add('markdown', '## 7. 학습 손실 곡선\n\n기존 trainer_state.json 로그 형식을 유지합니다.')
add('code', '''
import matplotlib.pyplot as plt
state = json.loads((Path(OUTPUT_DIR)/'trainer_state.json').read_text())
hist = [h for h in state['log_history'] if 'loss' in h]
if hist:
    plt.figure(figsize=(8.5,4.4))
    plt.plot([h['epoch'] for h in hist],[h['loss'] for h in hist],'o-',ms=3)
    for e in sorted(SAVE_EPOCHS): plt.axvline(e,color='gray',ls=':')
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.title(_tag); plt.tight_layout(); plt.show()
else:
    print('logging_steps보다 학습이 짧아 손실 로그가 없습니다.')
''')
add('markdown', '## 8. 불확실성\n\n원본처럼 recall=1을 정답으로 봅니다. 답변 클래스 entropy로 오답을 판별하며 Phi/Figure 3와 직접 동등한 실험은 아닙니다.')
add('code', '''
from sklearn.metrics import roc_auc_score, roc_curve
# 평가 중단/런타임 재시작 이후에도 미정의 records 변수를 참조하지 않습니다.
result_path = Path(RESULT_JSON)
if not result_path.is_file():
    raise FileNotFoundError(f'평가 결과가 없습니다: {result_path}. 먼저 6절 테스트 평가 셀을 끝까지 실행하세요.')
records = json.loads(result_path.read_text(encoding='utf-8'))
if not records:
    raise ValueError('평가 결과가 비어 있습니다. 6절 테스트 평가를 다시 실행하세요.')
ent = np.array([r['entropy'] for r in records])
wrong = np.array([int(r['score'] != 1.0) for r in records])
if len(np.unique(wrong)) == 2:
    auc = roc_auc_score(wrong, ent)
    fpr,tpr,_ = roc_curve(wrong,ent)
    print(f'Answer-class entropy error AUC = {auc:.3f}')
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    axes[0].plot(fpr,tpr,label=f'AUC={auc:.3f}'); axes[0].plot([0,1],[0,1],'k--')
    axes[0].legend(); axes[0].set(xlabel='FPR',ylabel='TPR',title=MODEL_NAME)
    axes[1].boxplot([ent[wrong==0],ent[wrong==1]])
    axes[1].set_xticks([1,2],['correct','incorrect'])
    axes[1].set_ylabel('Answer-class entropy')
    plt.tight_layout()
    plt.savefig(str(Path(RESULT_JSON).with_suffix('.uncertainty.png')),dpi=150)
    plt.show()
else:
    print('정답/오답 중 한 클래스만 있어 AUC를 계산하지 않습니다.')
top5 = np.argsort(-ent)[:max(1,int(round(.05*len(ent))))]
print('top-5%:',len(top5),'오답 비율:',wrong[top5].mean())
for at in ('OPEN','CLOSED'):
    ix = [i for i,r in enumerate(records) if r['answer_type']==at]
    if ix and len(np.unique(wrong[ix]))==2:
        print(at,'AUC:',roc_auc_score(wrong[ix],ent[ix]))
''')
add('markdown', '## 9. 결과 다운로드\n\n결과·checkpoint를 ZIP으로 내려받습니다. 런타임 종료 전에 실행하세요.')
add('code', '''
import shutil
from google.colab import files
archive = shutil.make_archive('/content/clip_vqarad_results', 'zip', '/content/clip_vqarad')
files.download(archive)
''')
n = {'cells':cells,'metadata':copy.deepcopy(original['metadata']),'nbformat':4,'nbformat_minor':5}
n['metadata']['accelerator']='GPU'
n['metadata']['colab']={'name':'VQA_RAD_PubMedCLIP_BiomedCLIP.ipynb','provenance':[]}
out = Path('notebooks/VQA_RAD_PubMedCLIP_BiomedCLIP.ipynb')
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(n,ensure_ascii=False,indent=1),encoding='utf-8')
print(out)
