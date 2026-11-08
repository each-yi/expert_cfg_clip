"""CPU checks of biomed_vital_qlora with a tiny random LLaMA/CLIP (not a benchmark, no 13B download).

Usage: python tests/biomed_vital_cpu_smoke.py <tokenizer_dir> <vqa_rad_json_dir> <work_dir>
  tokenizer_dir: BioMed-VITAL tokenizer files (tokenizer.model, tokenizer_config.json, special_tokens_map.json,
                 added_tokens.json) from mao1207/BioMed-VITAL-models@c93f260
  vqa_rad_json_dir: trainset.json / testset.json of the original QCR bundle (SHA-256 checked)
Images are replaced by synthetic ones; the 4-bit quantisation (CUDA only) is not exercised.
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import biomed_vital_qlora as bv  # noqa: E402

tok_dir, json_dir, work = map(Path, sys.argv[1:4])
if work.exists():
    shutil.rmtree(work)
rad_dir = work / 'rad_data'
(rad_dir / 'images').mkdir(parents=True)
for name in ('trainset.json', 'testset.json'):
    shutil.copyfile(json_dir / name, rad_dir / name)

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel, LlamaConfig, LlamaForCausalLM  # noqa: E402

for name in {r['image_name'] for f in ('trainset.json', 'testset.json') for r in json.loads((rad_dir / f).read_text(encoding='utf-8'))}:
    Image.new('L', (300 + len(name), 280), color=len(name) * 7 % 255).save(rad_dir / 'images' / name)

# ---- data: original split, validation by image ----
splits = bv.load_data('rad', work, work / 'cache', rad_dir=rad_dir)
summary = bv.data_summary(splits)
print('data', summary)
assert summary['test'] == {'questions': 451, 'images': 203, 'open': 179, 'closed': 272}
assert summary['train']['questions'] + summary['validation']['questions'] == 1797
assert not {r['image_name'] for r in splits['train']} & {r['image_name'] for r in splits['validation']}
assert all(r['phrase_type'] != 'frame' for v in splits.values() for r in v)

# ---- text format ----
tok = AutoTokenizer.from_pretrained(tok_dir, use_fast=False)
assert tok.convert_tokens_to_ids(['<im_patch>', '<im_start>', '<im_end>']) == [32000, 32001, 32002] and tok.pad_token_id == 0
ids, labels = bv.build_train_example(tok, 'Is there a rib fracture?', 'No', 576, 2048)
assert ids.count(32000) == 576 and ids.count(32001) == 1 and ids.count(32002) == 1
supervised = tok.decode([t for t, l in zip(ids, labels) if l != bv.IGNORE_INDEX])
print('supervised tokens:', repr(supervised))
assert 'No' in supervised and 'rib' not in supervised and '<im_patch>' not in supervised
prompt = bv.build_eval_prompt('Is there a rib fracture?', 576)
assert prompt.startswith(bv.SYSTEM + '###Human: What are the key differences') and prompt.endswith('<im_end>###')
assert prompt.count('###') == 4
assert bv.cut_output(' Assistant: No\n### Human: x') == 'Assistant: No'
assert bv.cut_output('### Assistant: No\n### Human: x') == 'Assistant: No'
assert bv.cut_output('\n###\n###Assistant: yes###') == 'Assistant: yes'

# ---- metrics (LLaVA-Med run_eval.py semantics) ----
assert bv.paper_score('CLOSED', 'Assistant: No, there is none', 'no') == 1.0
assert bv.paper_score('CLOSED', 'It is unclear', 'no') == 0.0
assert bv.paper_score('OPEN', 'the left lung and the heart', 'Left lung') == 1.0
assert abs(bv.paper_score('OPEN', 'right lung', 'left lung') - 0.5) < 1e-9
assert bv.extract_answer('Assistant: Left lung') == 'Left lung'
assert abs(bv.token_f1('right lung', 'left lung') - 0.5) < 1e-9
rows = [dict(qid=1, answer='no', answer_type='CLOSED'), dict(qid=2, answer='left lung', answer_type='OPEN')]
m, _ = bv.score(rows, {'1': 'Assistant: No', '2': 'Assistant: the left lung'})
print('score', m)
assert m['paper'] == {'open': 100.0, 'closed': 100.0, 'overall': 100.0}
assert m['unified']['closed_acc'] == 100.0 and m['unified']['open_em'] == 100.0

# ---- tiny model with the real geometry of the image path (336 / 14 -> 576 patches) ----
torch.manual_seed(0)
llm = LlamaForCausalLM(LlamaConfig(vocab_size=32003, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=4096,
                                   pad_token_id=0, bos_token_id=1, eos_token_id=2))
llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
llm.enable_input_require_grads()
vision = CLIPVisionModel(CLIPVisionConfig(image_size=336, patch_size=14, hidden_size=32, intermediate_size=64,
                                          num_hidden_layers=2, num_attention_heads=2))
processor = CLIPImageProcessor(size={'shortest_edge': 336}, crop_size={'height': 336, 'width': 336})
cfg = bv.config_for(dict(lrs=[1e-3, 3e-3], epochs=2, select_epochs=1, global_batch=4, micro_batch=2, eval_batch=3,
                         val_max_new_tokens=4, paper_max_new_tokens=4, save_every_steps=1, lora_r=4, lora_alpha=8))
proj = {'weight': torch.randn(64, 32) * 0.02, 'bias': torch.zeros(64)}
c = bv._assemble(cfg, tok, processor, vision, proj, llm, 'cpu')
print('trainable', sum(p.numel() for _, p in bv.trainable_named(c)))

spec = {'work_dir': str(work), 'cache_dir': str(work / 'cache'), 'datasets': ['rad'], 'rad_dir': str(rad_dir),
        'zero_shot': ['rad'], 'overrides': cfg, 'limit': {'train': 8, 'validation': 5, 'test': 6}}
s1 = bv.run_pipeline(spec, components=c)
print(json.dumps({k: v['unified'] for k, v in s1['results'].items()}, indent=1))
names = set(s1['results'])
chosen = bv.lr_tag(s1['lr_selection']['chosen_lr'])
assert {'zeroshot_rad_test_paper', f'rad_lr{chosen}_epoch2_test_paper', f'rad_lr{chosen}_epoch2_test_greedy',
        'rad_lr1e-3_epoch1_val_greedy', 'rad_lr3e-3_epoch1_val_greedy', f'rad_lr{chosen}_epoch2_val_greedy'} <= names, names
prog = {p['run']: p for p in bv.progress(work)}
assert prog[f'rad_lr{chosen}']['epoch_done'] == 2 and prog[f'rad_lr{chosen}']['global_step'] == '4/4'
other = [r for r in prog if r != f'rad_lr{chosen}'][0]
assert prog[other]['epoch_done'] == 1 and prog[other]['global_step'] == '2/4'
first = json.loads((work / 'results' / f'rad_lr{chosen}_epoch2_test_greedy.json').read_text(encoding='utf-8'))

# ---- resume: a fresh Run restores the saved parameters exactly; a rerun skips every finished stage ----
run = bv.Run(c, cfg, 'rad', s1['lr_selection']['chosen_lr'], {k: v[:8] for k, v in splits.items()},
             bv.pixel_cache(c, splits['train'][:8]), work / 'runs' / f'rad_lr{chosen}', log=lambda *_: None)
saved = torch.load(work / 'runs' / f'rad_lr{chosen}' / 'state.pt', weights_only=False)['params']
assert all(torch.equal(p.detach(), saved[n]) for n, p in bv.trainable_named(c)) and run.state['epoch'] == 2
bv._free(run)
s2 = bv.run_pipeline(spec, components=c, log=lambda *_: None)
assert s2['results'] == s1['results']
again = json.loads((work / 'results' / f'rad_lr{chosen}_epoch2_test_greedy.json').read_text(encoding='utf-8'))
assert again == first
backup = bv.backup_text(work)
assert backup.startswith(bv.BACKUP_BEGIN) and backup.endswith(bv.BACKUP_END)
print('backup', len(backup), 'chars; OK')
