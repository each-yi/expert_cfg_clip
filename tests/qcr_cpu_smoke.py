"""CPU smoke test of the patched official QCR code on a small subset (not a benchmark).

Usage: python tests/qcr_cpu_smoke.py <QCR_PubMedCLIP dir> <rad|slake> <PubMedCLIP .pth> [n_train n_test]
Runs the authors' train() for one epoch, which also calls evaluate_classifier on test.
"""
import os
import sys
from pathlib import Path

qcr, dataset, clip_path = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3]
n_train, n_test = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (40, 24)
sys.path[:0] = [str(qcr / 'main'), str(qcr / 'lib')]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(qcr)

import torch  # noqa: E402

_load = torch.load


def cpu_load(*args, **kwargs):
    # Checkpoints were saved on CUDA; also emulate the PyTorch>=2.6 weights_only=True default.
    kwargs.setdefault('map_location', 'cpu')
    kwargs.setdefault('weights_only', True)
    return _load(*args, **kwargs)


torch.load = cpu_load
import qcr_port  # noqa: E402
from config import cfg  # noqa: E402
from dataset import VQARADFeatureDataset, VQASLAKEFeatureDataset  # noqa: E402
from utils.create_dictionary import Dictionary  # noqa: E402
from BAN.multi_level_model import BAN_Model  # noqa: E402
from language.classify_question import classify_model  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from train import train  # noqa: E402

config = qcr_port.make_config(qcr, dataset, clip_path, 'cpu_smoke')
cfg.defrost()
cfg.merge_from_file(str(config))
cfg.TRAIN.N_EPOCH = 1
cfg.OUTPUT_DIR = str(qcr / 'output' / 'cpu_smoke')
cfg.freeze()
torch.manual_seed(cfg.SEED)
device = torch.device('cpu')
data_dir = cfg.DATASET.DATA_DIR
d = Dictionary.load_from_file(data_dir + '/dictionary.pkl')
Dataset = VQARADFeatureDataset if dataset == 'rad' else VQASLAKEFeatureDataset
train_ds, test_ds = Dataset('train', cfg, d, dataroot=data_dir), Dataset('test', cfg, d, dataroot=data_dir)
print('full sizes', len(train_ds), len(test_ds), 'close/open', train_ds.num_close_candidates, train_ds.num_open_candidates,
      'ans', train_ds.num_ans_candidates, 'ntoken', d.ntoken)
train_ds.entries, test_ds.entries = train_ds.entries[::len(train_ds) // n_train][:n_train], test_ds.entries[::len(test_ds) // n_test][:n_test]
train_loader = DataLoader(train_ds, cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_ds, cfg.TEST.BATCH_SIZE, shuffle=True, num_workers=0)
question_classify = classify_model(d.ntoken, os.path.join(data_dir, 'glove6b_init_300d.npy'))
if dataset == 'slake':
    state = torch.load('./saved_models/type_classifier_slake.pth', map_location='cpu', weights_only=True)['model_state']
else:
    state = torch.load('./saved_models/type_classifier.pth', map_location='cpu', weights_only=True)
question_classify.load_state_dict(state)
model = BAN_Model(train_ds, cfg, device)
print('v_dim', train_ds.v_dim, 'clip visual', type(model.clip.visual).__name__, 'dtype', next(model.clip.parameters()).dtype)
train(cfg, model, question_classify, train_loader, test_loader, train_ds.num_close_candidates, device)
log = Path(cfg.OUTPUT_DIR) / cfg.NAME / 'medVQA.log'
result = qcr_port.parse_log(log)
print('parsed', result)
assert result['epochs_logged'] == 1 and result['best'] is not None
