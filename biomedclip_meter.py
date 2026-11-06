"""BiomedCLIP + METER-style VQA on VQA-RAD / SLAKE: an APPROXIMATION of the BiomedCLIP paper row.

The BiomedCLIP paper (Table 11) fine-tunes "ViT-B/16-448-PMB/256" with METER but releases neither
the 448 checkpoint nor VQA code/hyper-parameters. This module therefore:
  * uses the public BiomedCLIP checkpoint (ViT-B/16-224 + PubMedBERT), pinned revision;
  * re-implements the METER co-attention fusion (zdou0830/METER: 6 BertCrossLayers per modality,
    CLS poolers, VQA head) with randomly initialised fusion, full fine-tuning of both encoders;
  * fills every unstated setting from M3AE (zhjohnchan/M3AE, a METER-based medical VQA code base):
    AdamW lr 5e-6, x5 cross-modal, x50 head, wd 0.01, 10% warm-up + linear decay, BCE x labels,
    answer set = all normalised answers of every split, max 32 question tokens,
    VQA-RAD batch 64 / 50 epochs, SLAKE batch 32 / 15 epochs;
  * evaluates like QCR/M3AE (test accuracy every epoch, best epoch reported) and also records the
    last epoch and, for SLAKE, the epoch chosen on the official validation split.
Data: the same VQA-RAD 3064/451 and SLAKE English files as the PubMedCLIP QCR reproduction (qcr_port).
"""
import base64
import hashlib
import io
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import qcr_port

BIOMEDCLIP = ('microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', '9f341de24bfb00180f1b847274256e9b65a3a32e',
              'open_clip_pytorch_model.bin', '52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be')
TEXT_CONFIG = ('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract', 'd673b8835373c6fa116d6d8006b33d48734e305d')
SLAKE_VALIDATION = ('BoKelvin/SLAKE', 'dataset', 'a9083ce6c34ac3ffb17671a605962924d8a8f9e9', 'validation.json',
                    '32b016440b0c3be11056a78a18eeab46333268407fbb6e6b32f9f4c2debc50f6')
qcr_port.ASSETS.setdefault('slake_validation', SLAKE_VALIDATION)
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

# BiomedCLIP paper Table 11 (METER row) = Expert-CFG Table 1 BiomedCLIP row: [open, closed, overall].
PAPER = {'rad': [67.60, 79.80, 75.20], 'slake': [82.50, 89.70, 85.40]}
COUNTS = {'rad': {'train': 3064, 'test': 451}, 'slake': {'train': 4919, 'validation': 1053, 'test': 1061}}
# M3AE task_finetune_vqa_vqa_rad / task_finetune_vqa_slack (+ METER defaults for the rest).
DEFAULTS = {
    'rad': dict(batch_size=64, epochs=50),
    'slake': dict(batch_size=32, epochs=15),
    'common': dict(learning_rate=5e-6, lr_mult_cross_modal=5, lr_mult_head=50, weight_decay=0.01, warmup=0.1,
                   betas=(0.9, 0.98), eps=1e-8, max_text_len=32, image_size=224, num_top_layer=6, hidden_size=768,
                   num_heads=12, mlp_ratio=4, drop_rate=0.1, micro_batch=16, seed=0, amp=True, grad_checkpointing=True, num_workers=2),
}

# ---- VQA answer normalisation (M3AE prepro/glossary.py normalize_word; same tables as QCR create_label.py)
CONTRACTIONS = {'aint': "ain't", 'arent': "aren't", 'cant': "can't", 'couldve': "could've", 'couldnt': "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've", 'didnt': "didn't", 'doesnt': "doesn't", 'dont': "don't", 'hadnt': "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've", 'hasnt': "hasn't", 'havent': "haven't", 'hed': "he'd", "hed've": "he'd've", "he'dve": "he'd've", 'hes': "he's", 'howd': "how'd", 'howll': "how'll", 'hows': "how's", "Id've": "I'd've", "I'dve": "I'd've", 'Im': "I'm", 'Ive': "I've", 'isnt': "isn't", 'itd': "it'd", "itd've": "it'd've", "it'dve": "it'd've", 'itll': "it'll", "let's": "let's", 'maam': "ma'am", 'mightnt': "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", 'mightve': "might've", 'mustnt': "mustn't", 'mustve': "must've", 'neednt': "needn't", 'notve': "not've", 'oclock': "o'clock", 'oughtnt': "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", 'shant': "shan't", "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's", 'shouldve': "should've", 'shouldnt': "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", "somebody'd": 'somebodyd', "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", 'somebodyll': "somebody'll", 'somebodys': "somebody's", 'someoned': "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", 'someonell': "someone'll", 'someones': "someone's", 'somethingd': "something'd", "somethingd've": "something'd've", "something'dve": "something'd've", 'somethingll': "something'll", 'thats': "that's", 'thered': "there'd", "thered've": "there'd've", "there'dve": "there'd've", 'therere': "there're", 'theres': "there's", 'theyd': "they'd", "theyd've": "they'd've", "they'dve": "they'd've", 'theyll': "they'll", 'theyre': "they're", 'theyve': "they've", 'twas': "'twas", 'wasnt': "wasn't", "wed've": "we'd've", "we'dve": "we'd've", 'weve': "we've", 'werent': "weren't", 'whatll': "what'll", 'whatre': "what're", 'whats': "what's", 'whatve': "what've", 'whens': "when's", 'whered': "where'd", 'wheres': "where's", 'whereve': "where've", 'whod': "who'd", "whod've": "who'd've", "who'dve": "who'd've", 'wholl': "who'll", 'whos': "who's", 'whove': "who've", 'whyll': "why'll", 'whyre': "why're", 'whys': "why's", 'wont': "won't", 'wouldve': "would've", 'wouldnt': "wouldn't", "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've", 'yall': "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", 'youd': "you'd", "youd've": "you'd've", "you'dve": "you'd've", 'youll': "you'll", 'youre': "you're", 'youve': "you've"}
MANUAL_MAP = {'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'}
ARTICLES = ['a', 'an', 'the']
PUNCT = [';', '/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']
_PERIOD_STRIP = re.compile(r'(?!<=\d)(\.)(?!\d)')
_COMMA_STRIP = re.compile(r'(\d)(\,)(\d)')


def normalize_answer(token):
    token = str(token).lower()
    _token = token
    for p in PUNCT:
        if (p + ' ' in token or ' ' + p in token) or (re.search(_COMMA_STRIP, token) is not None):
            _token = _token.replace(p, '')
        else:
            _token = _token.replace(p, ' ')
    token = _PERIOD_STRIP.sub('', _token, re.UNICODE)
    words = [MANUAL_MAP.get(w, w) for w in token.lower().split()]
    words = [CONTRACTIONS.get(w, w) for w in words if w not in ARTICLES]
    return ' '.join(words).replace(',', '')


# ---- data -----------------------------------------------------------------------------------------
def load_data(dataset, work_dir, cache_dir):
    """Rows with original question/answer/answer_type; the files are those of the QCR reproduction."""
    work_dir, cache_dir = Path(work_dir), Path(cache_dir)
    splits = {}
    if dataset == 'rad':
        repo = qcr_port._clone_original_qcr(work_dir)
        data = repo / 'data'
        for name in ('trainset.json', 'testset.json'):
            if qcr_port.sha256(data / name) != qcr_port.RAD_SHA256[name]:
                raise ValueError('VQA-RAD split file changed: ' + name)
        for split in ('train', 'test'):
            rows = json.loads((data / f'{split}set.json').read_text(encoding='utf-8'))
            splits[split] = [dict(qid=r['qid'], question=r['question'], answer=str(r['answer']),
                                  answer_type=r['answer_type'], image=str(data / 'images' / r['image_name'])) for r in rows]
    elif dataset == 'slake':
        root = work_dir / 'slake'
        if not (root / 'imgs').exists():
            root.mkdir(parents=True, exist_ok=True)
            qcr_port.safe_extract(qcr_port.fetch('slake_imgs', cache_dir), root)
        for split, key in (('train', 'slake_train'), ('validation', 'slake_validation'), ('test', 'slake_test')):
            rows = json.loads(Path(qcr_port.fetch(key, cache_dir)).read_text(encoding='utf-8'))
            splits[split] = [dict(qid=r['qid'], question=r['question'], answer=str(r['answer']),
                                  answer_type=r['answer_type'], image=str(root / 'imgs' / r['img_name']))
                             for r in rows if r['q_lang'] == 'en']
    else:
        raise ValueError('dataset must be rad or slake')
    counts = {k: len(v) for k, v in splits.items()}
    if counts != COUNTS[dataset]:
        raise ValueError(f'{dataset}: expected {COUNTS[dataset]}, got {counts}')
    for rows in splits.values():
        for r in rows:
            kind = r['answer_type'].strip().upper()
            if kind not in ('OPEN', 'CLOSED'):
                raise ValueError(f'Bad answer_type {r["answer_type"]!r}')
            r['answer_type'], r['label_text'] = kind, normalize_answer(r['answer'])
            if not Path(r['image']).exists():
                raise FileNotFoundError(r['image'])
    # M3AE make_arrow_vqa: label set = every normalised answer of every split, in first-seen order.
    labels = list(dict.fromkeys(r['label_text'] for split in splits for r in splits[split]))
    return splits, labels


# ---- model ------------------------------------------------------------------------------------------
def build_model(num_labels, cfg, weights_path, text_config_dir):
    import torch
    import timm
    from torch import nn
    from torch.nn import functional as F
    from transformers import BertConfig, BertModel
    from torch.utils.checkpoint import checkpoint

    hs, heads, drop = cfg['hidden_size'], cfg['num_heads'], cfg['drop_rate']

    class Attention(nn.Module):  # METER/BERT BertAttention: MHA + dense + dropout + residual LayerNorm
        def __init__(self):
            super().__init__()
            self.query, self.key, self.value = nn.Linear(hs, hs), nn.Linear(hs, hs), nn.Linear(hs, hs)
            self.dense, self.dropout, self.LayerNorm = nn.Linear(hs, hs), nn.Dropout(drop), nn.LayerNorm(hs, eps=1e-12)

        def forward(self, x, context, mask):
            b, n, _ = x.shape
            split = lambda t: t.view(b, -1, heads, hs // heads).transpose(1, 2)
            out = F.scaled_dot_product_attention(split(self.query(x)), split(self.key(context)), split(self.value(context)),
                                                 attn_mask=mask, dropout_p=drop if self.training else 0.0)
            return self.LayerNorm(self.dropout(self.dense(out.transpose(1, 2).reshape(b, n, hs))) + x)

    class BertCrossLayer(nn.Module):  # METER meter/modules/bert_model.py: self-attn -> cross-attn -> FFN
        def __init__(self):
            super().__init__()
            self.attention, self.crossattention = Attention(), Attention()
            self.intermediate = nn.Linear(hs, hs * cfg['mlp_ratio'])
            self.output, self.out_dropout = nn.Linear(hs * cfg['mlp_ratio'], hs), nn.Dropout(drop)
            self.out_norm = nn.LayerNorm(hs, eps=1e-12)

        def forward(self, x, y, x_mask, y_mask):
            x = self.crossattention(self.attention(x, x, x_mask), y, y_mask)
            return self.out_norm(self.out_dropout(self.output(F.gelu(self.intermediate(x)))) + x)

    class Pooler(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(hs, hs)

        def forward(self, x):
            return torch.tanh(self.dense(x[:, 0]))

    class BiomedCLIPMeter(nn.Module):
        def __init__(self):
            super().__init__()
            state = torch.load(weights_path, map_location='cpu', weights_only=True)
            self.vision = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0, global_pool='')
            vision_state = {k[len('visual.trunk.'):]: v for k, v in state.items() if k.startswith('visual.trunk.')}
            self.vision.load_state_dict(vision_state, strict=True)
            if cfg['image_size'] != 224:  # METER-style fine-tuning at higher resolution: resample pos. embeddings
                self.vision.set_input_size(img_size=(cfg['image_size'], cfg['image_size']))
            config = BertConfig.from_pretrained(text_config_dir)
            self.text = BertModel(config, add_pooling_layer=False)
            text_state = {k[len('text.transformer.'):]: v for k, v in state.items()
                          if k.startswith('text.transformer.') and not k.endswith('position_ids')}
            missing, unexpected = self.text.load_state_dict(text_state, strict=False)
            if unexpected or [k for k in missing if not k.endswith('position_ids')]:
                raise ValueError(f'BiomedCLIP text weights mismatch: missing={missing}, unexpected={unexpected}')
            del state
            self.cross_modal_text_transform = nn.Linear(config.hidden_size, hs)
            self.cross_modal_image_transform = nn.Linear(self.vision.num_features, hs)
            self.token_type_embeddings = nn.Embedding(2, hs)
            self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer() for _ in range(cfg['num_top_layer'])])
            self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer() for _ in range(cfg['num_top_layer'])])
            self.cross_modal_text_pooler, self.cross_modal_image_pooler = Pooler(), Pooler()
            self.vqa_classifier = nn.Sequential(nn.Linear(hs * 2, hs * 2), nn.LayerNorm(hs * 2), nn.GELU(),
                                                nn.Linear(hs * 2, num_labels))
            for name, module in self.named_children():  # METER init_weights for the new modules
                if name not in ('vision', 'text'):
                    module.apply(self._init)
            if cfg['grad_checkpointing']:
                self.vision.set_grad_checkpointing(True)
                self.text.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

        @staticmethod
        def _init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_(); module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        def forward(self, pixel_values, input_ids, attention_mask):
            text = self.text(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            image = self.vision.forward_features(pixel_values)
            text = self.cross_modal_text_transform(text) + self.token_type_embeddings.weight[0]
            image = self.cross_modal_image_transform(image) + self.token_type_embeddings.weight[1]
            # Boolean SDPA masks: True = attend. Images have no padding.
            text_mask = attention_mask[:, None, None, :].bool()
            image_mask = None
            for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
                def step(x, y, text_layer=text_layer, image_layer=image_layer):
                    return text_layer(x, y, text_mask, image_mask), image_layer(y, x, image_mask, text_mask)
                if cfg['grad_checkpointing'] and self.training:
                    text, image = checkpoint(step, text, image, use_reentrant=False)
                else:
                    text, image = step(text, image)
            return self.vqa_classifier(torch.cat([self.cross_modal_text_pooler(text), self.cross_modal_image_pooler(image)], -1))

    return BiomedCLIPMeter()


def param_groups(model, cfg):
    """METER set_schedule: x lr_mult_head for the VQA head, x lr_mult_cross_modal for 'cross_modal', no decay on bias/norm."""
    no_decay = ('bias', 'LayerNorm.bias', 'LayerNorm.weight', 'norm.bias', 'norm.weight', 'norm1.bias', 'norm1.weight',
                'norm2.bias', 'norm2.weight', 'out_norm.weight', 'out_norm.bias')
    groups = {}
    for name, p in model.named_parameters():
        mult = cfg['lr_mult_head'] if 'vqa_classifier' in name else cfg['lr_mult_cross_modal'] if 'cross_modal' in name else 1
        decay = 0.0 if any(nd in name for nd in no_decay) else cfg['weight_decay']
        groups.setdefault((mult, decay), []).append(p)
    return [{'params': ps, 'lr': cfg['learning_rate'] * m, 'weight_decay': wd} for (m, wd), ps in groups.items()]


# ---- training ---------------------------------------------------------------------------------------
def fit(dataset, work_dir, cache_dir, run_index=0, overrides=None, device='cuda', limit=None, micro_batch=None):
    """Train + evaluate one run; resumable per epoch.

    overrides: protocol changes (marks the run full_protocol=False). micro_batch: memory only, the effective
    batch is kept by gradient accumulation. limit=(n_train, n_eval) subsamples for smoke tests only.
    """
    import numpy as np
    import torch
    from PIL import Image
    from huggingface_hub import hf_hub_download, snapshot_download
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from transformers import AutoTokenizer, get_polynomial_decay_schedule_with_warmup

    cfg = config_for(dataset, overrides)
    if micro_batch:
        cfg['micro_batch'] = micro_batch
    full_protocol = not overrides and limit is None
    work_dir, cache_dir = Path(work_dir), Path(cache_dir)
    out = run_dir(work_dir, dataset, run_index, overrides, limit)
    out.mkdir(parents=True, exist_ok=True)
    seed = cfg['seed'] + run_index
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    splits, labels = load_data(dataset, work_dir, cache_dir)
    if limit:
        splits = {k: v[::max(1, len(v) // (limit[0] if k == 'train' else limit[1]))][:(limit[0] if k == 'train' else limit[1])] for k, v in splits.items()}
    ans2label = {a: i for i, a in enumerate(labels)}
    repo, revision, filename, expected = BIOMEDCLIP
    weights = hf_hub_download(repo, filename, revision=revision, cache_dir=str(cache_dir))
    if qcr_port.sha256(weights) != expected:
        raise ValueError('BiomedCLIP checkpoint hash mismatch')
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision, cache_dir=str(cache_dir))
    text_config_dir = snapshot_download(TEXT_CONFIG[0], revision=TEXT_CONFIG[1], allow_patterns=['config.json'], cache_dir=str(cache_dir))
    model = build_model(len(labels), cfg, weights, text_config_dir).to(device)

    size = cfg['image_size']  # METER "clip" transform: resize (bicubic) + centre crop + CLIP normalisation, no augmentation
    preprocess = transforms.Compose([transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
                                     transforms.CenterCrop(size), transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)])

    class Rows(Dataset):
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, i):
            with Image.open(self.rows[i]['image']) as im:
                return preprocess(im.convert('RGB')), i

    def collate_for(rows):
        def collate(batch):
            pixels, idx = zip(*batch)
            items = [rows[i] for i in idx]
            tok = tokenizer([r['question'] for r in items], padding='longest', truncation=True,
                            max_length=cfg['max_text_len'], return_tensors='pt')
            return torch.stack(pixels), tok['input_ids'], tok['attention_mask'], torch.tensor([ans2label[r['label_text']] for r in items]), items
        return collate

    micro = min(cfg['micro_batch'], cfg['batch_size'])
    accumulation = max(1, cfg['batch_size'] // micro)
    loaders = {k: DataLoader(Rows(v), batch_size=micro if k == 'train' else 32, shuffle=k == 'train', num_workers=cfg['num_workers'],
                             collate_fn=collate_for(v), pin_memory=device == 'cuda', drop_last=False) for k, v in splits.items()}
    optimizer = torch.optim.AdamW(param_groups(model, cfg), betas=tuple(cfg['betas']), eps=cfg['eps'])
    steps = math.ceil(len(loaders['train']) / accumulation) * cfg['epochs']
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, int(steps * cfg['warmup']), steps, lr_end=0, power=1)
    use_amp = cfg['amp'] and device == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    truncated = sum(len(tokenizer(r['question'])['input_ids']) > cfg['max_text_len'] for v in splits.values() for r in v)

    @torch.no_grad()
    def evaluate(split):
        model.eval()
        hit = Counter(); total = Counter(); predictions = []
        for pixels, ids, mask, target, items in loaders[split]:
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                logits = model(pixels.to(device), ids.to(device), mask.to(device))
            pred = logits.float().argmax(-1).cpu()
            for p, t, r in zip(pred.tolist(), target.tolist(), items):
                for group in (r['answer_type'], 'ALL'):
                    total[group] += 1; hit[group] += int(p == t)
                predictions.append({'qid': r['qid'], 'answer_type': r['answer_type'], 'answer': r['answer'], 'prediction': labels[p], 'correct': p == t})
        acc = {k.lower().replace('all', 'overall'): 100.0 * hit[k] / total[k] if total[k] else 0.0 for k in ('OPEN', 'CLOSED', 'ALL')}
        return {'open': acc['open'], 'closed': acc['closed'], 'overall': acc['overall']}, predictions

    state_path, history = out / 'last.pt', []
    start = 0
    if state_path.exists():  # only our own checkpoint from an earlier epoch of this run
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler'])
        start, history = state['epoch'], state['history']
        torch.set_rng_state(state['rng']); random.setstate(state['py_rng']); np.random.set_state(state['np_rng'])
        if device == 'cuda': torch.cuda.set_rng_state_all(state['cuda_rng'])
        print('resumed after epoch', start)
    for epoch in range(start, cfg['epochs']):
        model.train(); optimizer.zero_grad(set_to_none=True); loss_sum = seen = hits = 0
        batches = len(loaders['train'])
        for step, (pixels, ids, mask, target, items) in enumerate(loaders['train']):
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                logits = model(pixels.to(device), ids.to(device), mask.to(device))
            targets = F.one_hot(target.to(device), len(labels)).float()  # M3AE get_score() = 1.0
            loss = F.binary_cross_entropy_with_logits(logits.float(), targets) * targets.shape[1]
            window = min(accumulation, batches - (step // accumulation) * accumulation)
            scaler.scale(loss / window).backward()
            if (step + 1) % accumulation == 0 or step + 1 == batches:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step()
            loss_sum += float(loss) * len(items); seen += len(items)
            hits += int((logits.detach().argmax(-1).cpu() == target).sum())  # train mode (dropout on), like QCR Train_Acc
        row = {'epoch': epoch, 'train_loss': loss_sum / seen, 'train_acc': 100.0 * hits / seen, 'lr': scheduler.get_last_lr()[0]}
        for split in ('validation', 'test'):
            if split in loaders:
                row[split], predictions = evaluate(split)
                if split == 'test':
                    (out / f'test_predictions_epoch{epoch}.json').write_text(json.dumps(predictions), encoding='utf-8')
        history.append(row)
        print(json.dumps(row), flush=True)
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                 'scaler': scaler.state_dict(), 'epoch': epoch + 1, 'history': history, 'rng': torch.get_rng_state(),
                 'py_rng': random.getstate(), 'np_rng': np.random.get_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if device == 'cuda' else None}
        torch.save(state, out / 'last.tmp'); (out / 'last.tmp').replace(state_path)
        _write_json(out / 'history.json', history)  # atomic: the monitor cell reads it while training runs
    result = summarize_history(history)
    result.update(dataset=dataset, run=run_index, config=cfg, full_protocol=full_protocol and len(history) == cfg['epochs'],
                  num_labels=len(labels), truncated_questions=truncated, counts={k: len(v) for k, v in splits.items()},
                  paper_open_closed_overall=PAPER[dataset], output_dir=str(out),
                  weights=list(BIOMEDCLIP[:3]), deviations=DEVIATIONS)
    _write_json(out / 'result.json', result)
    return result


def config_for(dataset, overrides=None):
    return {**DEFAULTS['common'], **DEFAULTS[dataset], **(overrides or {})}


def run_dir(work_dir, dataset, run_index=0, overrides=None, limit=None):
    """Output folder of one run: runs/biomedclip_meter_<dataset>_run<i>/full, or quick_<overrides> if anything changed."""
    tag = 'full' if not overrides and limit is None else 'quick_' + '_'.join(f'{k}{v}' for k, v in sorted((overrides or {}).items()))
    return Path(work_dir) / 'runs' / f'biomedclip_meter_{dataset}_run{run_index}' / tag


def _write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    tmp.replace(path)


# ---- background runner -----------------------------------------------------------------------------------
# Training runs in a separate process (like the QCR reproduction, where main/main.py kept training after the
# notebook cell lost its output), so the notebook stays usable for monitoring cells while it trains.
DONE_MARK, START_MARK = 'ALL_JOBS_DONE', '=== trainer start'


def job_key(job):
    return job['dataset'], job['run'], json.dumps(job.get('overrides') or {}, sort_keys=True)


def run_jobs(jobs, work_dir, cache_dir, micro_batch=None):
    """Run jobs in order; each finished result is appended to work_dir/results.json and finished jobs are skipped."""
    results_path = Path(work_dir) / 'results.json'
    for job in jobs:
        results = json.loads(results_path.read_text(encoding='utf-8')) if results_path.exists() else []
        if any(job_key(r) == job_key(job) for r in results):
            print('skip finished', job_key(job), flush=True)
            continue
        print('\n=== start', job_key(job), time.strftime('%Y-%m-%d %H:%M:%S'), '===', flush=True)
        result = fit(job['dataset'], work_dir, cache_dir, run_index=job['run'], overrides=job.get('overrides') or None,
                     micro_batch=micro_batch)
        result['overrides'] = job.get('overrides') or {}
        results.append(result)
        _write_json(results_path, results)
        print('RESULT', json.dumps({k: result[k] for k in ('dataset', 'run', 'best_test_epoch', 'last_epoch', 'full_protocol')}), flush=True)


def _pid_alive(pid):
    """Linux /proc check; a finished child that was not reaped yet (zombie) counts as not running."""
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
        cmdline = Path(f'/proc/{pid}/cmdline').read_bytes()
    except OSError:
        return False
    return stat.rsplit(')', 1)[1].split()[0] != 'Z' and b'biomedclip_meter' in cmdline


def trainer_pid(work_dir):
    path = Path(work_dir) / 'train.pid'
    if path.exists():
        pid = int(path.read_text().strip())
        if _pid_alive(pid):
            return pid
    return None


def start_trainer(work_dir, cache_dir, jobs, micro_batch=None):
    """Start run_jobs in a background process (log: work_dir/train.log). Does nothing if one is already running."""
    work_dir = Path(work_dir)
    pid = trainer_pid(work_dir)
    if pid:
        print('trainer already running, pid', pid)
        return pid
    spec = work_dir / 'jobs.json'
    _write_json(spec, {'work_dir': str(work_dir), 'cache_dir': str(cache_dir), 'micro_batch': micro_batch, 'jobs': jobs})
    with open(work_dir / 'train.log', 'a', encoding='utf-8') as log:
        log.write(f'\n{START_MARK} {time.strftime("%Y-%m-%d %H:%M:%S")} ===\n')
        log.flush()
        proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), str(spec)], cwd=str(Path(__file__).resolve().parent),
                                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                                env=dict(os.environ, PYTHONUNBUFFERED='1'))
    (work_dir / 'train.pid').write_text(str(proc.pid))
    print('trainer started, pid', proc.pid, '| log', work_dir / 'train.log')
    return proc.pid


def stop_trainer(work_dir):
    import signal
    pid = trainer_pid(work_dir)
    if pid:
        os.killpg(pid, signal.SIGTERM)
        print('sent SIGTERM to', pid, '(the next start resumes from the last finished epoch)')
    else:
        print('no trainer running')


def trainer_status(work_dir, tail_lines=15):
    """running / finished (DONE_MARK after the latest start) / failed, plus the last log lines."""
    log = Path(work_dir) / 'train.log'
    text = log.read_text(encoding='utf-8', errors='replace') if log.exists() else ''
    latest = text.rsplit(START_MARK, 1)[-1] if START_MARK in text else ''
    state = 'running' if trainer_pid(work_dir) else 'finished' if DONE_MARK in latest else 'failed' if latest else 'not started'
    return {'state': state, 'tail': '\n'.join(latest.strip().splitlines()[-tail_lines:])}


def progress(work_dir, jobs):
    """Per job: epochs done, best-so-far and latest test overall, minutes since the last epoch was written."""
    rows = []
    for job in jobs:
        overrides = job.get('overrides') or None
        out = run_dir(work_dir, job['dataset'], job['run'], overrides)
        hist_path = out / 'history.json'
        hist = json.loads(hist_path.read_text(encoding='utf-8')) if hist_path.exists() else []
        best = summarize_history(hist)['best_test_epoch'] if hist else None
        rows.append({'dataset': job['dataset'], 'run': job['run'], 'overrides': overrides or {},
                     'epochs': f"{len(hist)}/{config_for(job['dataset'], overrides)['epochs']}",
                     'best_overall': round(best['overall'], 2) if best else None, 'best_epoch': best['epoch'] if best else None,
                     'last_overall': round(hist[-1]['test']['overall'], 2) if hist else None,
                     'last_train_acc': round(hist[-1]['train_acc'], 2) if hist and 'train_acc' in hist[-1] else None,
                     'min_since_update': round((time.time() - hist_path.stat().st_mtime) / 60, 1) if hist else None})
    return rows


# ---- backup printed into the notebook output (read back with extract_notebook_backup.py) -------------------
BACKUP_BEGIN, BACKUP_END = '===QCR_BACKUP_BEGIN===', '===QCR_BACKUP_END==='


def backup_text(work_dir, max_log_bytes=1 << 20):
    """Small zip (results, histories, predictions of the best/last/val-selected epochs, log tail) as base64 text."""
    work_dir = Path(work_dir)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name in ('results.json', 'summary.json', 'jobs.json'):
            if (work_dir / name).exists():
                z.write(work_dir / name, name)
        if (work_dir / 'train.log').exists():
            z.writestr('train.log', (work_dir / 'train.log').read_bytes()[-max_log_bytes:])
        for hist_path in sorted(work_dir.glob('runs/*/*/history.json')):
            out = hist_path.parent
            rel = out.relative_to(work_dir).as_posix()
            hist = json.loads(hist_path.read_text(encoding='utf-8'))
            z.write(hist_path, f'{rel}/history.json')
            if (out / 'result.json').exists():
                z.write(out / 'result.json', f'{rel}/result.json')
            if hist:
                s = summarize_history(hist)
                epochs = {s['best_test_epoch']['epoch'], hist[-1]['epoch']} | ({s['validation_selected']['epoch']} if 'validation_selected' in s else set())
                for e in sorted(epochs):
                    p = out / f'test_predictions_epoch{e}.json'
                    if p.exists():
                        z.write(p, f'{rel}/{p.name}')
    data = buf.getvalue()
    b64 = base64.b64encode(data).decode('ascii')
    lines = [b64[i:i + 100] for i in range(0, len(b64), 100)]
    return '\n'.join([BACKUP_BEGIN, f'biomedclip_backup.zip sha256={hashlib.sha256(data).hexdigest()}', *lines, BACKUP_END])


def summarize_history(history):
    best = None
    for row in history:  # QCR/M3AE: first epoch with strictly higher test overall accuracy
        if best is None or row['test']['overall'] > best['test']['overall']:
            best = row
    result = {'epochs_logged': len(history), 'best_test_epoch': best and {'epoch': best['epoch'], **best['test']},
              'last_epoch': history[-1]['test'] if history else None}
    if history and 'validation' in history[0]:
        chosen = max(history, key=lambda r: (r['validation']['overall'], -r['epoch']))
        result['validation_selected'] = {'epoch': chosen['epoch'], **chosen['test']}
    return result


def summarize(results):
    import statistics
    table = {}
    for dataset in PAPER:
        rows = [r for r in results if r['dataset'] == dataset]
        if not rows:
            continue
        entry = {'runs': len(rows), 'full_protocol': all(r['full_protocol'] for r in rows), 'paper_open_closed_overall': PAPER[dataset]}
        for protocol in ('best_test_epoch', 'last_epoch', 'validation_selected'):
            if all(protocol in r for r in rows):
                entry[protocol] = {k: statistics.mean(r[protocol][k] for r in rows) for k in ('open', 'closed', 'overall')}
                if len(rows) > 1:
                    entry[protocol]['overall_std'] = statistics.stdev(r[protocol]['overall'] for r in rows)
        table[dataset] = entry
    return table


DEVIATIONS = [
    'Public 224px BiomedCLIP checkpoint; the paper used an unreleased ViT-B/16-448 variant',
    'METER fusion re-implemented and randomly initialised; the paper gives no fusion details beyond METER',
    'Fine-tuning hyper-parameters, answer set and preprocessing taken from M3AE (METER-based medical VQA); the paper states none',
    'Model selection: best test epoch as in QCR/M3AE (paper does not say); last epoch and SLAKE validation-selected epoch also reported',
    'Open/Closed split by the ground-truth answer_type (VQA-RAD test 179/272); the PubMedCLIP QCR code splits by its '
    'question-type classifier prediction (logged 180/271), so only Overall shares the same denominator',
]


if __name__ == '__main__':  # background entry point used by start_trainer
    spec = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    run_jobs(spec['jobs'], spec['work_dir'], spec['cache_dir'], spec.get('micro_batch'))
    print(DONE_MARK, time.strftime('%Y-%m-%d %H:%M:%S'), flush=True)

