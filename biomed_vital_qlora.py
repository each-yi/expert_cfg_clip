"""BioMed-VITAL (13B) downstream VQA fine-tuning with QLoRA: an APPROXIMATION of the paper row.

What is public: only the instruction-tuned checkpoint (HF mao1207/BioMed-VITAL-models, LLaVA-Med v1
architecture: LLaMA-13B + one Linear mm_projector + CLIP ViT-L/14-336, <im_patch> tokens) and the authors'
inference script (eval/model_vqa_med.py). No VQA fine-tuned weights, no fine-tuning code. The paper says
"fine-tuned for 15 epochs with a learning rate of 2e-5" and "Following the LLaVA-Med approach".

This module therefore follows the LLaVA-Med v1.0.0 code for everything the paper leaves out:
  * training text / label masking: llava/train/train.py preprocess() (v0 path, "### Human: ... ### Assistant: ...");
  * image tokens: <im_start> + 576 x <im_patch> + <im_end>, patch embeddings replaced by projected CLIP
    features (hidden_states[-2], CLS dropped), as LlavaLlamaModel.forward;
  * inference prompt and decoding: BioMed-VITAL eval/model_vqa_med.py (conv "default" = conv_v1_2,
    do_sample=True, temperature 0.7, max_new_tokens 1024, stop at "###");
  * paper metrics: LLaVA-Med llava/eval/run_eval.py (closed: yes/no hit, open: token recall);
and replaces full fine-tuning by QLoRA (LLaVA v1.5 task-LoRA recipe: r 128, alpha 256, dropout 0.05,
all LLM linear layers, projector trained at 2e-5; 4-bit NF4 double-quant, bf16 compute).

Protocol (agreed with the professor): LoRA lr in {2e-5, 1e-4, 2e-4}, everything else identical. Every
run uses a 3-epoch cosine schedule (LLaVA-Med v1 command: num_train_epochs 3). Each lr is trained for
1 epoch on VQA-RAD train minus a held-out validation split; the lr with the best validation score is
continued (same optimizer/scheduler state) to epoch 3, so that run is identical to a 3-epoch run.
SLAKE is then trained 3 epochs with the chosen lr. Test sets are evaluated once, at the end.
VQA-RAD = the original release split (1,797 / 451; MEVF 'frame' questions removed), as in the paper.
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
import biomedclip_meter  # normalize_answer (VQA/M3AE/LLaVA-Med normalisation) and the SLAKE validation asset

MODEL = ('mao1207/BioMed-VITAL-models', 'c93f2607fdf20b16f3874e8fd5183ddffe705a9c')
MODEL_PATTERNS = ['config.json', 'generation_config.json', 'tokenizer.model', 'tokenizer_config.json',
                  'special_tokens_map.json', 'added_tokens.json', 'pytorch_model.bin.index.json', 'pytorch_model-*.bin']
VISION = ('openai/clip-vit-large-patch14-336', 'ce19dc912ca5cd21c8a653c79e251e808ccabcd1')

# BioMed-VITAL paper Table 4 (13B, 60K) = Expert-CFG Table 1: [open, closed, overall].
PAPER = {'rad': [64.88, 84.55, 76.74], 'slake': [87.82, 86.54, 87.32]}
SOURCE_COUNTS = {'rad': {'train': 1797, 'test': 451}, 'slake': {'train': 4919, 'validation': 1053, 'test': 1061}}

DEFAULTS = dict(
    epochs=3, select_epochs=1, lrs=[2e-5, 1e-4, 2e-4], projector_lr=2e-5,
    global_batch=64, micro_batch=4, weight_decay=0.0, warmup_ratio=0.03, max_grad_norm=1.0, max_length=2048,
    lora_r=128, lora_alpha=256, lora_dropout=0.05, seed=0, rad_val_fraction=0.1,
    eval_batch=8, val_max_new_tokens=128, paper_max_new_tokens=1024, temperature=0.7, save_every_steps=10,
)

# ---- LLaVA-Med v1.0.0 text format -------------------------------------------------------------------------
IGNORE_INDEX = -100
IMAGE_TOKEN, IM_PATCH, IM_START, IM_END = '<image>', '<im_patch>', '<im_start>', '<im_end>'
SEP = '###'
# llava/conversation.py conv_v1_2 (= conv_templates["default"], used by BioMed-VITAL eval/model_vqa_med.py)
SYSTEM = ("A chat between a curious human and an artificial intelligence assistant. "
          "The assistant gives helpful, detailed, and polite answers to the human's questions.")
FEWSHOT = (
    ('Human', 'What are the key differences between renewable and non-renewable energy sources?'),
    ('Assistant',
     'Renewable energy sources are those that can be replenished naturally in a relatively '
     'short amount of time, such as solar, wind, hydro, geothermal, and biomass. '
     'Non-renewable energy sources, on the other hand, are finite and will eventually be '
     'depleted, such as coal, oil, and natural gas. Here are some key differences between '
     'renewable and non-renewable energy sources:\n'
     '1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable '
     'energy sources are finite and will eventually run out.\n'
     '2. Environmental impact: Renewable energy sources have a much lower environmental impact '
     'than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, '
     'and other negative effects.\n'
     '3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically '
     'have lower operational costs than non-renewable sources.\n'
     '4. Reliability: Renewable energy sources are often more reliable and can be used in more remote '
     'locations than non-renewable sources.\n'
     '5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different '
     'situations and needs, while non-renewable sources are more rigid and inflexible.\n'
     '6. Sustainability: Renewable energy sources are more sustainable over the long term, while '
     'non-renewable sources are not, and their depletion can lead to economic and social instability.\n'),
)


def image_placeholder(n_patches):
    return IM_START + IM_PATCH * n_patches + IM_END  # mm_use_im_start_end = true in the checkpoint config


def _n_tokens(tokenizer, text, max_length):
    """train.py _tokenize_fn: length = non-pad tokens of tokenizer(text) (BOS included, pad = <unk>)."""
    ids = tokenizer(text, max_length=max_length, truncation=True).input_ids
    return sum(1 for t in ids if t != tokenizer.pad_token_id)


def build_train_example(tokenizer, question, answer, n_patches, max_length):
    """LLaVA-Med v1 train.py: preprocess_multimodal + preprocess (v0: _add_speaker_and_signal, _mask_targets).

    The human turn is "<question>\\n<image>" so the image follows the question exactly as at inference
    (model_vqa_med.py appends the image tokens after the question).
    """
    human = f'{question}\n{IMAGE_TOKEN}'.replace(IMAGE_TOKEN, image_placeholder(n_patches))
    header = f'{SYSTEM}\n\n'
    sentences = [('human', f'### Human: {human}\n'), ('gpt', f'### Assistant: {answer}\n')]
    conversation = header + ''.join(s for _, s in sentences) + '### '
    input_ids = tokenizer(conversation, max_length=max_length, truncation=True).input_ids
    labels = list(input_ids)
    lens = [_n_tokens(tokenizer, header, max_length)] + [_n_tokens(tokenizer, s, max_length) for _, s in sentences]
    cur = lens[0]
    labels[:cur] = [IGNORE_INDEX] * min(cur, len(labels))
    for length, (speaker, _) in zip(lens[1:], sentences):
        if speaker == 'human':
            for i in range(cur + 2, min(cur + length, len(labels))):
                labels[i] = IGNORE_INDEX
        cur += length
    return input_ids, labels


def build_eval_prompt(question, n_patches):
    """BioMed-VITAL eval/model_vqa_med.py with conv_mode 'default' (conv_v1_2, SeparatorStyle.SINGLE)."""
    qs = question.replace(IMAGE_TOKEN, '').strip() + '\n' + image_placeholder(n_patches)
    ret = SYSTEM + SEP
    for role, message in FEWSHOT + (('Human', qs),):
        ret += role + ': ' + message + SEP
    return ret


def cut_output(text):
    """model_vqa_med.py: keep the text before the first conv.sep ('###'), stripped.

    The fine-tuned model was trained on "...<im_end>\\n### Assistant: a" but the eval prompt already ends in '###',
    so it may answer "### Assistant: a"; a leading separator is dropped first (as the simple-mode loop of
    model_vqa_med.py does), otherwise the answer would be cut to ''.
    """
    text = text.strip()
    while text.startswith(SEP):
        text = text[len(SEP):].strip()
    index = text.find(SEP)
    return (text if index < 0 else text[:index]).strip()


# ---- metrics -------------------------------------------------------------------------------------------------
normalize_word = biomedclip_meter.normalize_answer  # same tables as LLaVA-Med eval_metrics/glossary.py


def _split_counts(sentence):
    return Counter(sentence.lower().strip().split())  # eval_metrics/utils.py split_sentence(sentence, 1)


def llava_f1(candidate, reference):
    """LLaVA-Med eval_metrics calculate_f1score -> (f1, precision, recall), counting as the original."""
    candidate_words = _split_counts(normalize_word(candidate))
    reference_words = _split_counts(normalize_word(reference))
    tp = fp = fn = 0
    for word in set(candidate_words) | set(reference_words):
        if word in candidate_words and word in reference_words:
            tp += candidate_words[word]
        elif word in candidate_words:
            fp += candidate_words[word]
        else:
            fn += reference_words[word]
    if not candidate_words or not reference_words or tp == 0:
        return 0.0, 0.0, 0.0
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    return 2 * precision * recall / (precision + recall), precision, recall


def paper_score(answer_type, prediction, answer):
    """run_eval.py: closed = yes/no hit, open = token recall (the paper's 'recall' for open questions)."""
    gt, pred = normalize_word(str(answer).lower()), normalize_word(str(prediction).lower())
    if answer_type == 'CLOSED':
        return float(('yes' in pred or 'no' in pred) and gt in pred)
    return llava_f1(pred, gt)[2]


_ROLE_PREFIX = re.compile(r'^\s*(assistant|gpt)\s*:\s*', re.I)


def extract_answer(prediction):
    """For the unified metrics only: drop a leading role tag the model may emit ('Assistant: ...')."""
    return _ROLE_PREFIX.sub('', prediction).strip()


def token_f1(prediction, answer):
    p, g = normalize_word(prediction).split(), normalize_word(answer).split()
    common = sum((Counter(p) & Counter(g)).values())
    if not p or not g or common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def score(rows, predictions):
    """Paper protocol (run_eval.py) and the unified protocol (same normalisation for every model).

    Overall = mean over all questions, i.e. Open and Closed weighted by their counts, as in the paper tables.
    Open/Closed always follow the ground-truth answer_type.
    """
    by_type = {'OPEN': [], 'CLOSED': []}
    per_question = []
    for r in rows:
        pred = predictions[str(r['qid'])]
        clean = extract_answer(pred)
        em = float(normalize_word(clean) == normalize_word(r['answer']))
        item = {'qid': r['qid'], 'answer_type': r['answer_type'], 'answer': r['answer'], 'prediction': pred,
                'paper': paper_score(r['answer_type'], pred, r['answer']), 'em': em, 'f1': token_f1(clean, r['answer'])}
        by_type[r['answer_type']].append(item)
        per_question.append(item)
    mean = lambda xs: 100.0 * sum(xs) / len(xs) if xs else 0.0
    allq = by_type['OPEN'] + by_type['CLOSED']
    result = {
        'counts': {'open': len(by_type['OPEN']), 'closed': len(by_type['CLOSED'])},
        'paper': {'open': mean([i['paper'] for i in by_type['OPEN']]), 'closed': mean([i['paper'] for i in by_type['CLOSED']]),
                  'overall': mean([i['paper'] for i in allq])},
        'unified': {'closed_acc': mean([i['em'] for i in by_type['CLOSED']]), 'open_em': mean([i['em'] for i in by_type['OPEN']]),
                    'open_f1': mean([i['f1'] for i in by_type['OPEN']]), 'overall_em': mean([i['em'] for i in allq])},
    }
    return result, per_question


def selection_key(val_result):
    """Validation rule for choosing the lr: unified overall EM, then open token F1."""
    u = val_result['unified']
    return (round(u['overall_em'], 6), round(u['open_f1'], 6))


# ---- data ----------------------------------------------------------------------------------------------------
def _rows(records, image_root, image_key):
    rows = []
    for r in records:
        kind = str(r['answer_type']).strip().upper()
        if kind not in ('OPEN', 'CLOSED'):
            raise ValueError(f'Bad answer_type {r["answer_type"]!r}')
        rows.append(dict(qid=r['qid'], question=r['question'], answer=str(r['answer']), answer_type=kind,
                         image_name=r[image_key], image=str(Path(image_root) / r[image_key]), phrase_type=r.get('phrase_type')))
    return rows


def split_validation(rows, fraction, seed):
    """Hold out whole images (no image shared between train and validation)."""
    images = sorted({r['image_name'] for r in rows})
    random.Random(seed).shuffle(images)
    held = set(images[:max(1, round(len(images) * fraction))])
    return [r for r in rows if r['image_name'] not in held], [r for r in rows if r['image_name'] in held]


def load_data(dataset, work_dir, cache_dir, cfg=None, rad_dir=None):
    """VQA-RAD: original split (MEVF/QCR files with phrase_type 'frame' removed) + image-level validation.
    SLAKE: BoKelvin/SLAKE English train/validation/test. Files pinned and hashed (qcr_port)."""
    cfg = {**DEFAULTS, **(cfg or {})}
    work_dir, cache_dir = Path(work_dir), Path(cache_dir)
    if dataset == 'rad':
        data = Path(rad_dir) if rad_dir else qcr_port._clone_original_qcr(work_dir) / 'data'
        for name in ('trainset.json', 'testset.json'):
            if qcr_port.sha256(data / name) != qcr_port.RAD_SHA256[name]:
                raise ValueError('VQA-RAD split file changed: ' + name)
        train = [r for r in json.loads((data / 'trainset.json').read_text(encoding='utf-8')) if r['phrase_type'] != 'frame']
        test = json.loads((data / 'testset.json').read_text(encoding='utf-8'))
        source = {'train': len(train), 'test': len(test)}
        train, test = _rows(train, data / 'images', 'image_name'), _rows(test, data / 'images', 'image_name')
        train, validation = split_validation(train, cfg['rad_val_fraction'], cfg['seed'])
        splits = {'train': train, 'validation': validation, 'test': test}
    elif dataset == 'slake':
        root = work_dir / 'slake'
        if not (root / 'imgs').exists():
            root.mkdir(parents=True, exist_ok=True)
            qcr_port.safe_extract(qcr_port.fetch('slake_imgs', cache_dir), root)
        splits = {}
        for split, key in (('train', 'slake_train'), ('validation', 'slake_validation'), ('test', 'slake_test')):
            records = [r for r in json.loads(Path(qcr_port.fetch(key, cache_dir)).read_text(encoding='utf-8')) if r['q_lang'] == 'en']
            splits[split] = _rows(records, root / 'imgs', 'img_name')
        source = {k: len(v) for k, v in splits.items()}
    else:
        raise ValueError('dataset must be rad or slake')
    if source != SOURCE_COUNTS[dataset]:
        raise ValueError(f'{dataset}: expected {SOURCE_COUNTS[dataset]}, got {source}')
    if dataset == 'rad' and any(r['phrase_type'] == 'frame' for r in splits['test']):
        raise ValueError('VQA-RAD test unexpectedly contains frame questions')
    for rows in splits.values():
        for r in rows:
            if not Path(r['image']).exists():
                raise FileNotFoundError(r['image'])
    return splits


def data_summary(splits):
    return {k: {'questions': len(v), 'images': len({r['image_name'] for r in v}),
                'open': sum(r['answer_type'] == 'OPEN' for r in v), 'closed': sum(r['answer_type'] == 'CLOSED' for r in v)}
            for k, v in splits.items()}


# ---- model ---------------------------------------------------------------------------------------------------
LLAMA_KEYS = ('vocab_size', 'hidden_size', 'intermediate_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads',
              'hidden_act', 'max_position_embeddings', 'rms_norm_eps', 'initializer_range', 'bos_token_id', 'eos_token_id',
              'pad_token_id', 'tie_word_embeddings', 'rope_scaling', 'pretraining_tp')
LORA_TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


def find_all_linear_names(model):
    """LLaVA train.py find_all_linear_names (LLM linear layers, without lm_head / multimodal parts)."""
    import torch
    names = set()
    for name, module in model.named_modules():
        if any(k in name for k in ('mm_projector', 'vision_tower', 'vision_resampler')):
            continue
        if isinstance(module, torch.nn.Linear):
            names.add(name.split('.')[-1])
    names.discard('lm_head')
    return sorted(names)


def download_model(cache_dir):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(MODEL[0], revision=MODEL[1], allow_patterns=MODEL_PATTERNS, cache_dir=str(cache_dir)))


class Components:
    """tokenizer, image processor, frozen CLIP tower, trainable projector, LLM with LoRA."""

    def __init__(self, tokenizer, image_processor, vision, projector, model, n_patches, device, initial_projector):
        self.tokenizer, self.image_processor, self.vision, self.projector, self.model = tokenizer, image_processor, vision, projector, model
        self.n_patches, self.device, self.initial_projector = n_patches, device, initial_projector
        self.patch_id = tokenizer.convert_tokens_to_ids(IM_PATCH)
        self.start_id, self.end_id = tokenizer.convert_tokens_to_ids([IM_START, IM_END])


def build_components(cfg, cache_dir, device='cuda'):
    """4-bit NF4 LLaMA-13B (LLaVA train.py BitsAndBytesConfig) + LoRA, fp32 projector, fp16 CLIP tower."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor, CLIPVisionModel, LlamaConfig, LlamaForCausalLM

    model_dir = download_model(cache_dir)
    raw = json.loads((model_dir / 'config.json').read_text(encoding='utf-8'))
    assert raw['mm_use_im_start_end'] and raw['mm_vision_select_layer'] == -2 and raw['mm_vision_tower'] == VISION[0]
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=['lm_head'])
    llm = LlamaForCausalLM.from_pretrained(model_dir, config=LlamaConfig(**{k: raw.get(k) for k in LLAMA_KEYS}),
                                           quantization_config=bnb, torch_dtype=torch.bfloat16, device_map={'': 0},
                                           low_cpu_mem_usage=True, attn_implementation='sdpa')
    llm = prepare_model_for_kbit_training(llm, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False})
    index = json.loads((model_dir / 'pytorch_model.bin.index.json').read_text(encoding='utf-8'))['weight_map']
    shard = torch.load(model_dir / index['model.mm_projector.weight'], map_location='cpu', weights_only=True, mmap=True)
    projector_state = {'weight': shard['model.mm_projector.weight'].float().clone(), 'bias': shard['model.mm_projector.bias'].float().clone()}
    del shard
    vision = CLIPVisionModel.from_pretrained(VISION[0], revision=VISION[1], torch_dtype=torch.float16, cache_dir=str(cache_dir))
    image_processor = CLIPImageProcessor.from_pretrained(VISION[0], revision=VISION[1], cache_dir=str(cache_dir))
    return _assemble(cfg, tokenizer, image_processor, vision, projector_state, llm, device)


def _assemble(cfg, tokenizer, image_processor, vision, projector_state, llm, device):
    import torch
    from peft import LoraConfig, get_peft_model
    vision = vision.to(device).eval().requires_grad_(False)
    n_patches = (vision.config.image_size // vision.config.patch_size) ** 2
    projector = torch.nn.Linear(vision.config.hidden_size, llm.config.hidden_size).to(device)
    projector.load_state_dict(projector_state)
    targets = find_all_linear_names(llm)
    if targets != sorted(LORA_TARGETS):
        raise ValueError(f'unexpected LoRA targets {targets}')
    lora = LoraConfig(r=cfg['lora_r'], lora_alpha=cfg['lora_alpha'], target_modules=targets, lora_dropout=cfg['lora_dropout'],
                      bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(llm, lora)
    for tok in (IM_PATCH, IM_START, IM_END):
        if tokenizer.convert_tokens_to_ids(tok) in (None, tokenizer.unk_token_id):
            raise ValueError('tokenizer misses ' + tok)
    return Components(tokenizer, image_processor, vision, projector, model, n_patches, device,
                      {k: v.detach().cpu().clone() for k, v in projector.state_dict().items()})


def reset_trainable(c, seed):
    """Fresh run: LoRA re-initialised (A kaiming, B zero, as peft does) and the checkpoint's projector."""
    import torch
    from peft.tuners.lora import LoraLayer
    torch.manual_seed(seed)
    for module in c.model.modules():
        if isinstance(module, LoraLayer):
            module.reset_lora_parameters('default', True)
    c.projector.load_state_dict(c.initial_projector)


def trainable_named(c):
    params = [(f'lora.{n}', p) for n, p in c.model.named_parameters() if p.requires_grad]
    params += [(f'projector.{n}', p) for n, p in c.projector.named_parameters()]
    return params


def pixel_cache(c, rows):
    import torch
    from PIL import Image
    cache = {}
    for r in rows:
        if r['image_name'] not in cache:
            with Image.open(r['image']) as im:
                cache[r['image_name']] = c.image_processor.preprocess(im.convert('RGB'), return_tensors='pt')['pixel_values'][0].to(torch.float16)
    return cache


def embed(c, input_ids, pixel_values):
    """LlavaLlamaModel.forward: word embeddings with the <im_patch> positions replaced by projected CLIP patch features."""
    import torch
    embeds = c.model.get_input_embeddings()(input_ids)
    with torch.no_grad():
        feats = c.vision(pixel_values.to(c.device, torch.float16), output_hidden_states=True).hidden_states[-2][:, 1:]
    image = c.projector(feats.float())
    mask = input_ids == c.patch_id
    if not bool((mask.sum(1) == c.n_patches).all()):
        raise ValueError('every sample needs exactly one image (n_patches <im_patch> tokens)')
    embeds = embeds.clone()
    embeds[mask] = image.reshape(-1, image.shape[-1]).to(embeds.dtype)
    return embeds


def _autocast(c):
    import torch
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=c.device == 'cuda')


# ---- generation ------------------------------------------------------------------------------------------------
def generate(c, rows, pixels, mode, cfg, seed=0, log=print):
    """Batched (left-padded) generation with the eval prompt. mode 'paper': do_sample, T=0.7, 1024 new tokens;
    mode 'greedy': deterministic, used for validation (and as an extra test decoding)."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    tok = c.tokenizer

    class StopOnSep(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return torch.tensor([SEP in t for t in tok.batch_decode(input_ids, skip_special_tokens=True)], device=input_ids.device)

    kwargs = dict(do_sample=True, temperature=cfg['temperature'], max_new_tokens=cfg['paper_max_new_tokens']) if mode == 'paper' \
        else dict(do_sample=False, max_new_tokens=cfg['val_max_new_tokens'])
    c.model.eval()
    torch.manual_seed(seed)
    out, raw, start = {}, {}, time.time()
    for b in range(0, len(rows), cfg['eval_batch']):
        batch = rows[b:b + cfg['eval_batch']]
        prompts = [tok(build_eval_prompt(r['question'], c.n_patches)).input_ids for r in batch]
        width = max(map(len, prompts))
        ids = torch.tensor([[tok.pad_token_id] * (width - len(p)) + p for p in prompts], device=c.device)
        mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=c.device)
        pix = torch.stack([pixels[r['image_name']] for r in batch])
        with torch.inference_mode(), _autocast(c):
            gen = c.model.generate(inputs_embeds=embed(c, ids, pix), attention_mask=mask, pad_token_id=tok.pad_token_id,
                                   eos_token_id=tok.eos_token_id, use_cache=True,
                                   stopping_criteria=StoppingCriteriaList([StopOnSep()]), **kwargs)
        for r, text in zip(batch, tok.batch_decode(gen, skip_special_tokens=True)):
            out[str(r['qid'])] = cut_output(text)
            raw[str(r['qid'])] = text
        if b == 0 or (b // cfg['eval_batch']) % 10 == 0:
            done = min(b + cfg['eval_batch'], len(rows))
            log(f'  generate[{mode}] {done}/{len(rows)}  {(time.time() - start) / done:.2f}s/question  '
                f'eta {(time.time() - start) / done * (len(rows) - done) / 60:.1f} min')
    return out, raw


# ---- training --------------------------------------------------------------------------------------------------
def _collate(c, examples, pixels):
    import torch
    width = max(len(e['input_ids']) for e in examples)
    pad = c.tokenizer.pad_token_id
    ids = torch.tensor([e['input_ids'] + [pad] * (width - len(e['input_ids'])) for e in examples], device=c.device)
    labels = torch.tensor([e['labels'] + [IGNORE_INDEX] * (width - len(e['labels'])) for e in examples], device=c.device)
    mask = torch.tensor([[1] * len(e['input_ids']) + [0] * (width - len(e['input_ids'])) for e in examples], device=c.device)
    return ids, labels, mask, torch.stack([pixels[e['image_name']] for e in examples])


def _write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    tmp.replace(path)


class Run:
    """One fine-tuning run (dataset, lr) with a fixed cfg['epochs'] cosine schedule; resumable at optimizer steps."""

    def __init__(self, c, cfg, dataset, lr, splits, pixels, out_dir, log=print):
        import torch
        from transformers import get_cosine_schedule_with_warmup
        self.c, self.cfg, self.dataset, self.lr, self.splits, self.pixels, self.log = c, cfg, dataset, lr, splits, pixels, log
        self.out = Path(out_dir)
        self.examples = []
        for r in splits['train']:
            ids, labels = build_train_example(c.tokenizer, r['question'], r['answer'], c.n_patches, cfg['max_length'])
            self.examples.append({'input_ids': ids, 'labels': labels, 'image_name': r['image_name']})
        self.steps_per_epoch = math.ceil(len(self.examples) / cfg['global_batch'])
        self.total_steps = self.steps_per_epoch * cfg['epochs']
        reset_trainable(c, cfg['seed'])
        self.params = trainable_named(c)
        lora = [p for n, p in self.params if n.startswith('lora.')]
        proj = [p for n, p in self.params if n.startswith('projector.')]
        self.optimizer = torch.optim.AdamW([{'params': lora, 'lr': lr, 'weight_decay': cfg['weight_decay']},
                                            {'params': proj, 'lr': cfg['projector_lr'], 'weight_decay': cfg['weight_decay']}],
                                           betas=(0.9, 0.999), eps=1e-8)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, math.ceil(self.total_steps * cfg['warmup_ratio']), self.total_steps)
        self.state = {'epoch': 0, 'step_in_epoch': 0, 'global_step': 0, 'history': [], 'losses': []}
        self._resume()

    def _resume(self):
        import numpy as np
        import torch
        path = self.out / 'state.pt'
        if not path.exists():
            return
        saved = torch.load(path, map_location='cpu', weights_only=False)
        named = dict(self.params)
        for n, t in saved['params'].items():
            named[n].data.copy_(t.to(named[n].device, named[n].dtype))
        self.optimizer.load_state_dict(saved['optimizer'])
        self.scheduler.load_state_dict(saved['scheduler'])
        self.state = saved['state']
        torch.set_rng_state(saved['rng']); random.setstate(saved['py_rng']); np.random.set_state(saved['np_rng'])
        if saved.get('cuda_rng') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        self.log(f'resumed {self.out.name}: epoch {self.state["epoch"]}, step {self.state["step_in_epoch"]}/{self.steps_per_epoch}')

    def save(self):
        import numpy as np
        import torch
        self.out.mkdir(parents=True, exist_ok=True)
        saved = {'params': {n: p.detach().cpu() for n, p in self.params}, 'optimizer': self.optimizer.state_dict(),
                 'scheduler': self.scheduler.state_dict(), 'state': self.state, 'rng': torch.get_rng_state(),
                 'py_rng': random.getstate(), 'np_rng': np.random.get_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        torch.save(saved, self.out / 'state.tmp')
        (self.out / 'state.tmp').replace(self.out / 'state.pt')
        _write_json(self.out / 'progress.json', {**self.state, 'steps_per_epoch': self.steps_per_epoch, 'total_steps': self.total_steps,
                                                 'lr': self.lr, 'dataset': self.dataset, 'updated': time.time()})

    def train_until(self, epoch_end, on_epoch_end=None):
        """Train until cfg epoch index `epoch_end` (1-based count of finished epochs) is reached."""
        import torch
        cfg, c = self.cfg, self.c
        micro = min(cfg['micro_batch'], cfg['global_batch'])
        while self.state['epoch'] < epoch_end:
            epoch = self.state['epoch']
            order = torch.randperm(len(self.examples), generator=torch.Generator().manual_seed(cfg['seed'] * 1000 + epoch)).tolist()
            c.model.train()
            t0, steps_timed = time.time(), 0
            for step in range(self.state['step_in_epoch'], self.steps_per_epoch):
                batch = [self.examples[i] for i in order[step * cfg['global_batch']:(step + 1) * cfg['global_batch']]]
                chunks = [batch[i:i + micro] for i in range(0, len(batch), micro)]
                self.optimizer.zero_grad(set_to_none=True)
                total = 0.0
                for chunk in chunks:  # HF Trainer: mean token loss per micro-batch / accumulation steps
                    ids, labels, mask, pix = _collate(c, chunk, self.pixels)
                    with _autocast(c):
                        loss = c.model(inputs_embeds=embed(c, ids, pix), attention_mask=mask, labels=labels).loss / len(chunks)
                    loss.backward()
                    total += float(loss)
                torch.nn.utils.clip_grad_norm_([p for _, p in self.params], cfg['max_grad_norm'])
                self.optimizer.step(); self.scheduler.step()
                self.state['step_in_epoch'] = step + 1
                self.state['global_step'] += 1
                self.state['losses'].append(round(total, 5))
                steps_timed += 1
                sec = (time.time() - t0) / steps_timed
                remaining = (self.steps_per_epoch - step - 1) + (epoch_end - epoch - 1) * self.steps_per_epoch
                self.log(f'{self.dataset} lr={self.lr:g} epoch {epoch + 1} step {step + 1}/{self.steps_per_epoch} '
                         f'loss {total:.4f} lr_now {self.scheduler.get_last_lr()[0]:.2e} {sec:.1f}s/step '
                         f'eta(train to epoch {epoch_end}) {remaining * sec / 60:.1f} min')
                if not math.isfinite(total):
                    raise FloatingPointError(f'non-finite loss at {self.dataset} lr={self.lr} step {self.state["global_step"]}')
                if self.state['step_in_epoch'] % cfg['save_every_steps'] == 0 and self.state['step_in_epoch'] < self.steps_per_epoch:
                    self.save()
            self.state['epoch'] += 1
            self.state['step_in_epoch'] = 0
            record = {'epoch': self.state['epoch'], 'mean_loss': sum(self.state['losses'][-self.steps_per_epoch:]) / self.steps_per_epoch}
            if on_epoch_end:
                record.update(on_epoch_end(self, self.state['epoch']) or {})
            self.state['history'].append(record)
            self.save()
            self.log(json.dumps({'run': self.out.name, **record}))


# ---- pipeline --------------------------------------------------------------------------------------------------
def config_for(overrides=None):
    return {**DEFAULTS, **(overrides or {})}


def lr_tag(lr):
    return f'{lr:.0e}'.replace('e-0', 'e-')


def _free(run):
    """Drop a finished run's optimizer state (~4 GB for r=128) before the next run is created."""
    import gc
    import torch
    run.optimizer = run.scheduler = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_split(c, cfg, splits, pixels, split, mode, results_dir, name, seed=0, log=print):
    path = Path(results_dir) / f'{name}.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))['metrics']
    rows = splits[split]
    preds, raw = generate(c, rows, pixels, mode, cfg, seed=seed, log=log)
    metrics, per_question = score(rows, preds)
    for item in per_question:
        item['raw_output'] = raw[str(item['qid'])]
    metrics.update(split=split, decoding=mode, questions=len(rows))
    _write_json(path, {'metrics': metrics, 'per_question': per_question})
    log(f'RESULT {name} ' + json.dumps({'paper': metrics['paper'], 'unified': metrics['unified']}))
    return metrics


def run_pipeline(spec, components=None, log=print):
    """Stages (each skipped when its result exists): zero-shot test, RAD lr sweep (1 epoch, validation), continue
    the chosen lr to epoch 3 + test, SLAKE 3 epochs with the chosen lr + test."""
    cfg = config_for(spec.get('overrides'))
    work, cache = Path(spec['work_dir']), Path(spec['cache_dir'])
    res = work / 'results'
    c = components or build_components(cfg, cache)
    data = {d: load_data(d, work, cache, cfg, rad_dir=spec.get('rad_dir')) for d in ('rad', 'slake') if d in spec['datasets']}
    if spec.get('limit'):  # smoke tests only
        n = spec['limit']
        data = {d: {k: v[:n[k]] for k, v in s.items()} for d, s in data.items()}
    for d, s in data.items():
        log(f'data {d}: ' + json.dumps(data_summary(s)))
    pixels = {d: pixel_cache(c, [r for v in s.values() for r in v]) for d, s in data.items()}
    test_modes = ['paper'] + (['greedy'] if spec.get('test_greedy', True) else [])

    for d in spec.get('zero_shot', []):
        if d in data:
            evaluate_split(c, cfg, data[d], pixels[d], 'test', 'paper', res, f'zeroshot_{d}_test_paper', log=log)

    if 'rad' in data:
        rad = data['rad']
        val_hook = lambda run, epoch: {'validation': evaluate_split(
            c, cfg, rad, pixels['rad'], 'validation', 'greedy', res, f'rad_lr{lr_tag(run.lr)}_epoch{epoch}_val_greedy', log=log)['unified']}
        sweep = {}
        for lr in cfg['lrs']:
            run = Run(c, cfg, 'rad', lr, rad, pixels['rad'], work / 'runs' / f'rad_lr{lr_tag(lr)}', log=log)
            run.train_until(cfg['select_epochs'], on_epoch_end=val_hook)
            sweep[lr] = evaluate_split(c, cfg, rad, pixels['rad'], 'validation', 'greedy', res,
                                       f'rad_lr{lr_tag(lr)}_epoch{cfg["select_epochs"]}_val_greedy', log=log)
            _free(run)
        chosen = spec.get('chosen_lr') or max(cfg['lrs'], key=lambda lr: (selection_key(sweep[lr]), -lr))
        _write_json(res / 'lr_selection.json', {'chosen_lr': chosen, 'manual': bool(spec.get('chosen_lr')),
                                                'rule': 'validation unified overall_em, then open_f1, then smaller lr',
                                                'validation_after_epochs': cfg['select_epochs'],
                                                'sweep': {lr_tag(lr): sweep[lr] for lr in cfg['lrs']}})
        log(f'chosen lr {chosen:g}')
        run = Run(c, cfg, 'rad', chosen, rad, pixels['rad'], work / 'runs' / f'rad_lr{lr_tag(chosen)}', log=log)
        run.train_until(cfg['epochs'], on_epoch_end=val_hook)
        for mode in test_modes:
            evaluate_split(c, cfg, rad, pixels['rad'], 'test', mode, res, f'rad_lr{lr_tag(chosen)}_epoch{cfg["epochs"]}_test_{mode}', log=log)
        _free(run)
    else:
        chosen = spec.get('chosen_lr')

    if 'slake' in data:
        if chosen is None:
            sel = res / 'lr_selection.json'
            chosen = json.loads(sel.read_text(encoding='utf-8'))['chosen_lr'] if sel.exists() else None
        if chosen is None:
            raise ValueError('SLAKE needs chosen_lr (run VQA-RAD first or set CHOSEN_LR)')
        slake = data['slake']
        run = Run(c, cfg, 'slake', chosen, slake, pixels['slake'], work / 'runs' / f'slake_lr{lr_tag(chosen)}', log=log)
        run.train_until(cfg['epochs'])
        for mode in test_modes:
            evaluate_split(c, cfg, slake, pixels['slake'], 'test', mode, res, f'slake_lr{lr_tag(chosen)}_epoch{cfg["epochs"]}_test_{mode}', log=log)
        if spec.get('slake_validation'):
            evaluate_split(c, cfg, slake, pixels['slake'], 'validation', 'greedy', res, f'slake_lr{lr_tag(chosen)}_epoch{cfg["epochs"]}_val_greedy', log=log)
    summary = summarize(work)
    _write_json(work / 'summary.json', summary)
    return summary


def summarize(work_dir):
    res = Path(work_dir) / 'results'
    rows = {}
    for path in sorted(res.glob('*.json')):
        if path.name == 'lr_selection.json':
            continue
        rows[path.stem] = json.loads(path.read_text(encoding='utf-8'))['metrics']
    sel = res / 'lr_selection.json'
    return {'results': rows, 'lr_selection': json.loads(sel.read_text(encoding='utf-8')) if sel.exists() else None,
            'paper_open_closed_overall': PAPER, 'deviations': DEVIATIONS, 'config': DEFAULTS}


# ---- background runner (same pattern as biomedclip_meter) ------------------------------------------------------------
DONE_MARK, START_MARK = 'ALL_STAGES_DONE', '=== pipeline start'


def _pid_alive(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
        cmdline = Path(f'/proc/{pid}/cmdline').read_bytes()
    except OSError:
        return False
    return stat.rsplit(')', 1)[1].split()[0] != 'Z' and b'biomed_vital_qlora' in cmdline


def trainer_pid(work_dir):
    path = Path(work_dir) / 'train.pid'
    if path.exists():
        pid = int(path.read_text().strip())
        if _pid_alive(pid):
            return pid
    return None


def start_trainer(spec):
    work = Path(spec['work_dir'])
    work.mkdir(parents=True, exist_ok=True)
    pid = trainer_pid(work)
    if pid:
        print('pipeline already running, pid', pid)
        return pid
    _write_json(work / 'spec.json', spec)
    with open(work / 'train.log', 'a', encoding='utf-8') as log:
        log.write(f'\n{START_MARK} {time.strftime("%Y-%m-%d %H:%M:%S")} ===\n')
        log.flush()
        proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), str(work / 'spec.json')],
                                cwd=str(Path(__file__).resolve().parent), stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED='1'))
    (work / 'train.pid').write_text(str(proc.pid))
    print('pipeline started, pid', proc.pid, '| log', work / 'train.log')
    return proc.pid


def stop_trainer(work_dir):
    import signal
    pid = trainer_pid(work_dir)
    if pid:
        os.killpg(pid, signal.SIGTERM)
        print('sent SIGTERM to', pid, '(restart resumes from the last saved step)')
    else:
        print('no pipeline running')


def trainer_status(work_dir, tail_lines=15):
    log = Path(work_dir) / 'train.log'
    text = log.read_text(encoding='utf-8', errors='replace') if log.exists() else ''
    latest = text.rsplit(START_MARK, 1)[-1] if START_MARK in text else ''
    state = 'running' if trainer_pid(work_dir) else 'finished' if DONE_MARK in latest else 'failed' if latest else 'not started'
    return {'state': state, 'tail': '\n'.join(latest.strip().splitlines()[-tail_lines:])}


def progress(work_dir):
    rows = []
    for p in sorted(Path(work_dir).glob('runs/*/progress.json')):
        s = json.loads(p.read_text(encoding='utf-8'))
        rows.append({'run': p.parent.name, 'epoch_done': s['epoch'], 'step_in_epoch': f"{s['step_in_epoch']}/{s['steps_per_epoch']}",
                     'global_step': f"{s['global_step']}/{s['total_steps']}", 'last_loss': s['losses'][-1] if s['losses'] else None,
                     'min_since_save': round((time.time() - s['updated']) / 60, 1)})
    return rows


BACKUP_BEGIN, BACKUP_END = '===QCR_BACKUP_BEGIN===', '===QCR_BACKUP_END==='


def backup_text(work_dir, max_log_bytes=1 << 20):
    """results/*.json (metrics + per-question predictions), summary, spec, run histories and log tail, base64 zip."""
    work = Path(work_dir)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name in ('summary.json', 'spec.json'):
            if (work / name).exists():
                z.write(work / name, name)
        for p in sorted(work.glob('results/*.json')) + sorted(work.glob('runs/*/progress.json')):
            z.write(p, p.relative_to(work).as_posix())
        if (work / 'train.log').exists():
            z.writestr('train.log', (work / 'train.log').read_bytes()[-max_log_bytes:])
    data = buf.getvalue()
    b64 = base64.b64encode(data).decode('ascii')
    return '\n'.join([BACKUP_BEGIN, f'biomed_vital_backup.zip sha256={hashlib.sha256(data).hexdigest()}',
                      *[b64[i:i + 100] for i in range(0, len(b64), 100)], BACKUP_END])


DEVIATIONS = [
    'QLoRA (4-bit NF4 LLaMA-13B + LoRA r128/alpha256 on all LLM linear layers, projector trained) instead of the '
    'full-model fine-tuning of LLaVA-Med / BioMed-VITAL (8 GPUs); LoRA recipe from LLaVA v1.5 finetune_task_lora.sh',
    '3 epochs (LLaVA-Med v1.0.0 fine-tuning command) instead of the paper\'s 15, because of compute',
    'LoRA lr chosen from {2e-5, 1e-4, 2e-4} on a VQA-RAD validation split after 1 epoch of the 3-epoch schedule; '
    'the paper\'s 2e-5 is a full fine-tuning lr. The projector keeps 2e-5. SLAKE reuses the VQA-RAD choice',
    'VQA-RAD validation = 10% of the training images (seed 0), so VQA-RAD is trained on ~90% of the 1,797 training '
    'questions; the paper has no validation split',
    'Fine-tuning text format, image tokens and label masking follow LLaVA-Med v1.0.0 train.py (v0); the authors\' '
    'downstream fine-tuning data/script are not public. Human turn = "<question>\\n<image>"',
    'Generated text: a leading "###" is dropped before cutting at the first "###" (training ends the human turn with '
    '"\\n### Assistant:", the eval prompt with "###", so the fine-tuned model can open with "###"); raw outputs are saved',
    'Validation uses greedy decoding (max 128 new tokens) for a deterministic choice; the test set is decoded with '
    'the authors\' settings (sampling, T=0.7, 1024 new tokens, seed 0) and additionally greedily',
    'Paper metrics = LLaVA-Med run_eval.py (closed: yes/no hit, open: token recall). Unified metrics = normalised '
    'exact match (closed accuracy, open EM) and token F1 after removing a leading "Assistant:" tag, Open/Closed by '
    'the ground-truth answer_type, for comparison with the classification baselines',
    'Generation is batched with left padding (bf16 autocast); the authors generate one question at a time in fp16',
]


if __name__ == '__main__':  # background entry point used by start_trainer
    spec = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    run_pipeline(spec)
    print(DONE_MARK, time.strftime('%Y-%m-%d %H:%M:%S'), flush=True)
