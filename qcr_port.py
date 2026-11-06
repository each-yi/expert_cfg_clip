"""Run the official PubMedCLIP QCR code on a current Colab GPU runtime.

The official code (sarahESL/PubMedCLIP, QCR_PubMedCLIP, pinned commit) is cloned and
only environment-compatibility edits are applied (see PATCHES). Model, loss, optimizer,
data preprocessing scripts and configs are the authors' own. Every external asset is
pinned to an exact revision (HF revision or git commit) and verified by SHA-256.
"""
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

QCR_REPO = 'https://github.com/sarahESL/PubMedCLIP.git'
QCR_COMMIT = 'bdecf3c5a21b68200a02dc05455b3882a23206a2'

# repo, repo_type, revision, filename, sha256
ASSETS = {
    'mevf_rad': ('aiozai/MEVF', 'dataset', '7b3bd1b1ef8965eb35ff61a1f6fada25377765ea', 'data_RAD.zip',
                 'b75d664c3dac7096a695e3a8a2b309769804f5add7e72b15e037ea9e5b839a9f'),
    'slake_imgs': ('BoKelvin/SLAKE', 'dataset', 'a9083ce6c34ac3ffb17671a605962924d8a8f9e9', 'imgs.zip',
                   '44eb7d9214e1ac5b7946e237b669401866df9114fd9a3618c5a84fbffcded0b0'),
    'slake_train': ('BoKelvin/SLAKE', 'dataset', 'a9083ce6c34ac3ffb17671a605962924d8a8f9e9', 'train.json',
                    '7a9fb81f9d7bb145c3d7e9bc8d3b9016ba18b6539ea5031a53bae8fd3cd991c3'),
    'slake_test': ('BoKelvin/SLAKE', 'dataset', 'a9083ce6c34ac3ffb17671a605962924d8a8f9e9', 'test.json',
                   '6be8f7b4c5a46cdbc713a5210a25b6ed5aa1fd1574c83cefb4f998131f17c2c3'),
    'RN50': ('sarahESL/PubMedCLIP', 'model', 'c5885a5c0ba8ac8d86341e57c8466e49783874c1', 'PubMedCLIP_RN50.pth',
             'bb8b2bfd7f6e38fb7fb1861d969e4df862776f605aede4ecb200d2e72f9d4c6a'),
    'ViT': ('sarahESL/PubMedCLIP', 'model', 'c5885a5c0ba8ac8d86341e57c8466e49783874c1', 'PubMedCLIP_ViT32.pth',
            '56d244ea2e547b59fd0357eb17d481240eeb21590e5bae2c91ea0a55de82d9c6'),
}

# Expert-CFG Table 1 PubMedCLIP rows = PubMedCLIP paper Table 1 (QCR + AE): [open, closed, overall].
RUNS = {
    'rad': {'encoder': 'RN50', 'config': 'qcr_pubmedclipRN50_ae_rad_16batchsize_withtfidf_nondeterministic.yaml',
            'paper': [60.10, 80.00, 72.10], 'counts': {'train': 3064, 'test': 451}},
    'slake': {'encoder': 'ViT', 'config': 'qcr_pubmedclipViT_ae_slake_16batchsize_withtfidf_nondeterministic.yaml',
              'paper': [78.40, 82.50, 80.10], 'counts': {'train': 4919, 'test': 1061}},
}

# The original QCR authors' repository ships the prepared VQA-RAD bundle (data/). PubMedCLIP's
# QCR code at QCR_COMMIT hard-codes its label space (487 = 56 closed + 431 open, open labels
# indexed locally) and its type classifier was trained on that 1177-word dictionary; the
# PubMedCLIP create_label.py/create_dictionary.py would instead produce 64+483 labels and
# 1259 words, which the code cannot load. Train/test JSON, imgid2idx, auto-encoder and
# images128 are byte-identical to the MEVF bundle on HF (aiozai/MEVF data_RAD.zip).
ORIGINAL_QCR_REPO = 'https://github.com/Awenbocc/med-vqa.git'
ORIGINAL_QCR_COMMIT = '0cca6811e38cf54aff6a7cce3442296d07875e64'
RAD_SHA256 = {
    'trainset.json': '3b0567673ae224b6489ca1bc4ce05cda642223d5cf4f17db8fbf7010c872e00f',
    'testset.json': '3d32a047592cb680e18054030b38b1a44d6b17105fc62d3088c8655d56a00ced',
    'imgid2idx.json': '4ee2f8ebc4adf3700c10307c0b3daac3edafaf08717b4d20eef423228e316257',
    'dictionary.pkl': '35a01367a5d5a4417d6f99c045bf8f3ccca51bb48da0e37e1aae38590cc40504',
    'glove6b_init_300d.npy': '7b70281a90e0b141d4655481718a99f9d3f09da23d2f10608e874173c1057bf1',
    'embed_tfidf_weights.pkl': '0bb3cbb06714f8987886abffe69e0d38bf312ce7b9406b2b804915a568fcfd27',
    'images128x128.pkl': '136cffd2c2222993c9cc7035ded0fa11d7ce1cae37ab0baaa951f85a002c6e6f',
    'pretrained_ae.pth': 'f2fb0fdc6d014fb49b85fcd2029902e14db9b42d21f3718d1a07d7082960bd05',
    'cache/close_label2ans.pkl': 'b50f9d24cb6a54599ffda7b371f3f42419be42aa5c58f9069d531cf7ffe9d863',
    'cache/open_label2ans.pkl': '965340e4b4e9ad152e7d055987f25cafe630666d70b6fee795ccc526d28a1f4a',
    'cache/trainval_ans2label.pkl': 'daaebd434b2d1e06d79aebf24d750adca76dcb1e4548ada83a16aee0848b6b77',
    'cache/trainval_label2ans.pkl': '22bd133cc95f2d2c25973db5e463e0c3a603c84ea4404bec1020299b8d587e83',
    'cache/train_openclose_target.pkl': '2b99f7ade3d38d0fb5da77a4af65f8afb505f1afb5657e2b23623126c602edf5',
    'cache/test_openclose_target.pkl': '4df33f0095ea260d91b6525d3318281c8313cebf28f0dd5c56d363635342c09b',
}

# Inserted into lib/utils/utils.py. Globals were audited with pickletools on the pinned file.
TFIDF_LOADER = '''string_classes = (str, bytes)
import pickle as _pickle
import types as _types
try:  # referenced by the PyTorch 0.4 pickle of embed_tfidf_weights.pkl
    import torch.nn.backends.thnn
except ImportError:
    import sys as _sys
    _thnn = _types.ModuleType('torch.nn.backends.thnn')
    _thnn._get_thnn_function_backend = lambda: None
    _sys.modules['torch.nn.backends.thnn'] = _thnn
_TFIDF_GLOBALS = {('collections', 'OrderedDict'), ('language_model', 'WordEmbedding'),
                  ('language.language_model', 'WordEmbedding'),  # file written by QCR itself (SLAKE)
                  ('__builtin__', 'set'), ('builtins', 'set'),
                  ('torch', 'FloatStorage'), ('torch._utils', '_rebuild_parameter'),
                  ('torch._utils', '_rebuild_tensor_v2'), ('torch.nn.backends.thnn', '_get_thnn_function_backend'),
                  ('torch.nn.modules.dropout', 'Dropout'), ('torch.nn.modules.sparse', 'Embedding')}
class _AllowlistUnpickler(_pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _TFIDF_GLOBALS:
            raise _pickle.UnpicklingError('Blocked global in embed_tfidf_weights.pkl: %s.%s' % (module, name))
        return super().find_class(module, name)
_tfidf_pickle = _types.ModuleType('qcr_allowlist_pickle')
_tfidf_pickle.Unpickler = _AllowlistUnpickler
_tfidf_pickle.load = lambda f, **kw: _AllowlistUnpickler(f, **kw).load()
'''

# (file, old, new, reason). Each `old` must occur exactly once in the pinned commit.
PATCHES = [
    ('lib/utils/utils.py', 'from torch._six import string_classes\n', TFIDF_LOADER,
     'torch._six was removed in PyTorch 2; also defines the allowlisted loader used below'),
    ('lib/utils/utils.py', 'collections.Mapping)', 'collections.abc.Mapping)', 'Python 3.10 removed collections.Mapping'),
    ('lib/utils/utils.py', 'collections.Sequence)', 'collections.abc.Sequence)', 'Python 3.10 removed collections.Sequence'),
    ('lib/utils/utils.py', 'w_emb = torch.load(f)\n',
     'w_emb = torch.load(f, weights_only=False, pickle_module=_tfidf_pickle)\n',
     'embed_tfidf_weights.pkl is a PyTorch-0.4 pickled nn.Module; PyTorch>=2.6 defaults to weights_only=True. '
     'Load it with an unpickler that only allows the audited classes'),
    ('lib/utils/utils.py', 'torch.sparse.FloatTensor(torch.LongTensor(inds), torch.FloatTensor(vals))',
     'torch.sparse_coo_tensor(torch.LongTensor(inds), torch.FloatTensor(vals))', 'legacy sparse constructor'),
    ('lib/utils/create_label.py', 'qa_pairs = train_qa_pairs.append(val_qa_pairs)\n    qa_pairs[\'answer\'] = qa_pairs[\'answer\'].apply(lambda x: str(x))\n\n    for id',
     'qa_pairs = pd.concat([train_qa_pairs, val_qa_pairs])\n    qa_pairs[\'answer\'] = qa_pairs[\'answer\'].apply(lambda x: str(x))\n\n    for id',
     'DataFrame.append was removed in pandas 2'),
    ('lib/utils/create_label.py', 'qa_pairs = train_qa_pairs.append(val_qa_pairs)\n    qa_pairs[\'answer\'] = qa_pairs[\'answer\'].apply(lambda x: str(x))\n    qa_pairs_open',
     'qa_pairs = pd.concat([train_qa_pairs, val_qa_pairs])\n    qa_pairs[\'answer\'] = qa_pairs[\'answer\'].apply(lambda x: str(x))\n    qa_pairs_open',
     'DataFrame.append was removed in pandas 2'),
    ('lib/utils/create_label.py', 'df = train_qa_pairs.append(val_qa_pairs)', 'df = pd.concat([train_qa_pairs, val_qa_pairs])',
     'DataFrame.append was removed in pandas 2'),
    ('lib/utils/create_img2idx.py', 'df = train_en.append(val_en)', 'df = pd.concat([train_en, val_en])',
     'DataFrame.append was removed in pandas 2'),
    ('lib/BAN/multi_level_model.py',
     "            self.clip, _ = clip.load(cfg.TRAIN.VISION.CLIP_VISION_ENCODER, jit=False)\n"
     "            if not cfg.TRAIN.VISION.CLIP_ORG:\n"
     "                checkpoint = torch.load(cfg.TRAIN.VISION.CLIP_PATH)\n"
     "                self.clip.load_state_dict(checkpoint['state_dict'])\n",
     "            if not cfg.TRAIN.VISION.CLIP_ORG:\n"
     "                checkpoint = torch.load(cfg.TRAIN.VISION.CLIP_PATH, map_location='cpu')\n"
     "                # Same as clip.load(name) + load_state_dict: build_model() casts to fp16 and the\n"
     "                # PubMedCLIP weights are copied into those fp16 parameters before .float().\n"
     "                self.clip = clip.model.build_model(checkpoint['state_dict']).to(device)\n"
     "                self.clip.load_state_dict(checkpoint['state_dict'])\n"
     "            else:\n"
     "                self.clip, _ = clip.load(cfg.TRAIN.VISION.CLIP_VISION_ENCODER, jit=False)\n",
     'clip.load(name) first downloads OpenAI weights (openaipublic.azureedge.net CDN) that are immediately '
     'overwritten; build the identical architecture from the PubMedCLIP state dict instead'),
    ('main/classifier.py', 'nn.Dropout(cfg.TRAIN.DROPOUT, inplace=True)', 'nn.Dropout(cfg.TRAIN.DROPOUT, inplace=False)',
     'current autograd rejects in-place dropout on the saved ReLU output; out-of-place dropout gives the same '
     'forward values and gradients (dropped units get zero gradient either way)'),
    ('main/train.py',
     "                utils.save_model(model_path, model, best_epoch, eval_score, open_score, close_score, optim)\n",
     "                utils.save_model(model_path, model, best_epoch, eval_score, open_score, close_score, optim)\n"
     "                if os.environ.get('QCR_KEEP_ALL_BEST', '0') != '1':\n"
     "                    for old in os.listdir(ckpt_path):\n"
     "                        if old.endswith('_best.pth') and old != os.path.basename(model_path):\n"
     "                            os.remove(os.path.join(ckpt_path, old))\n",
     'each improvement saves a ~1 GB checkpoint (CLIP + optimizer); keep only the newest one so 200 epochs fit '
     'on Colab disk (file handling only; QCR_KEEP_ALL_BEST=1 restores the original behaviour)'),
    ('main/main.py', '    torch.autograd.set_detect_anomaly(True)\n',
     '    torch.autograd.set_detect_anomaly(os.environ.get("QCR_DETECT_ANOMALY", "0") == "1")\n',
     'anomaly detection is a debugging aid (no numerical effect) that slows training several times; '
     'set QCR_DETECT_ANOMALY=1 to restore it'),
]


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def run(cmd, cwd=None, env=None):
    import os
    print('$', ' '.join(map(str, cmd)), flush=True)
    # The authors' scripts open GloVe/JSON with the platform default encoding.
    env = dict(env or os.environ, PYTHONUTF8='1', PYTHONUNBUFFERED='1')
    # Stream through Python so the output is visible in a notebook cell.
    with subprocess.Popen(list(map(str, cmd)), cwd=cwd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as proc:
        for line in proc.stdout:
            print(line, end='', flush=True)
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def fetch(key, cache_dir):
    from huggingface_hub import hf_hub_download
    repo, repo_type, revision, filename, expected = ASSETS[key]
    path = Path(hf_hub_download(repo, filename, repo_type=repo_type, revision=revision, cache_dir=str(cache_dir)))
    if expected and sha256(path) != expected:
        raise ValueError(f'SHA-256 mismatch for {repo}/{filename}@{revision}')
    return path


def safe_extract(zip_path, target):
    target = Path(target)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            name = PurePosixPath(member.filename)
            if name.is_absolute() or '..' in name.parts or '\\' in member.filename:
                raise ValueError('Unsafe archive member: ' + member.filename)
        archive.extractall(target)


def setup_code(work_dir, source=None):
    """Clone the pinned official repo (or copy a local clone) and apply PATCHES."""
    work_dir = Path(work_dir)
    repo_root = work_dir / 'PubMedCLIP'
    if not repo_root.exists():
        if source:
            shutil.copytree(source, repo_root, ignore=shutil.ignore_patterns('.git'))
        else:
            run(['git', 'clone', '--quiet', QCR_REPO, repo_root])
            run(['git', 'checkout', '--quiet', QCR_COMMIT], cwd=repo_root)
    if (repo_root / '.git').exists():
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo_root, capture_output=True, text=True, check=True).stdout.strip()
        if head != QCR_COMMIT:
            raise ValueError(f'Expected QCR commit {QCR_COMMIT}, found {head}')
    qcr = repo_root / 'QCR_PubMedCLIP'
    applied = []
    for rel, old, new, reason in PATCHES:
        path = qcr / rel
        text = path.read_text(encoding='utf-8')
        if text.count(old) == 1:
            path.write_text(text.replace(old, new), encoding='utf-8')
        elif not (text.count(old) == 0 and new in text):
            raise ValueError(f'Patch target not found exactly once in {rel}: {old[:60]!r}')
        applied.append({'file': rel, 'reason': reason})
    return qcr, applied


def _check_counts(dataset, data_dir):
    if dataset == 'rad':
        train = json.loads((data_dir / 'trainset.json').read_text(encoding='utf-8'))
        test = json.loads((data_dir / 'testset.json').read_text(encoding='utf-8'))
    else:
        train = [r for r in json.loads((data_dir / 'train.json').read_text(encoding='utf-8')) if r['q_lang'] == 'en']
        test = [r for r in json.loads((data_dir / 'test.json').read_text(encoding='utf-8')) if r['q_lang'] == 'en']
    counts = {'train': len(train), 'test': len(test)}
    if counts != RUNS[dataset]['counts']:
        raise ValueError(f'{dataset}: expected {RUNS[dataset]["counts"]}, got {counts}')
    return counts


def _clone_original_qcr(work_dir):
    repo = Path(work_dir) / 'med-vqa'
    if not repo.exists():
        run(['git', 'clone', '--quiet', ORIGINAL_QCR_REPO, repo])
    run(['git', 'checkout', '--quiet', ORIGINAL_QCR_COMMIT], cwd=repo)
    return repo


def _extract_member(zip_path, member, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive, archive.open(member) as src, open(target, 'wb') as dst:
        shutil.copyfileobj(src, dst, 1 << 22)


def prepare_data(qcr, dataset, cache_dir):
    """Create data/data_rad or data/data_slake as the QCR code at QCR_COMMIT expects."""
    qcr, cache_dir = Path(qcr), Path(cache_dir)
    data_root = qcr / 'data'
    rad_dir, slake_dir = data_root / 'data_rad', data_root / 'data_slake'
    utils_dir = qcr / 'lib' / 'utils'
    py = sys.executable
    marker = data_root / f'.{dataset}_ready'
    if marker.exists():
        return json.loads(marker.read_text(encoding='utf-8'))
    if dataset == 'rad':
        source = _clone_original_qcr(qcr.parent.parent) / 'data'
        if rad_dir.exists():
            shutil.rmtree(rad_dir)
        shutil.copytree(source, rad_dir, ignore=shutil.ignore_patterns('*.ipynb', '.ipynb_checkpoints'))
        for rel, expected in RAD_SHA256.items():
            if sha256(rad_dir / rel) != expected:
                raise ValueError(f'Original QCR RAD file changed: {rel}')
        counts = _check_counts('rad', rad_dir)
        # The only input missing from the bundle: CLIP images, made with PubMedCLIP's own script
        # exactly as in lib/utils/run.sh (250x250, 3 channels).
        run([py, 'create_resized_images.py', rad_dir / 'imgid2idx.json', rad_dir / 'images',
             250, rad_dir / 'images250x250.pkl', 3], cwd=utils_dir)
        assets = {'original_qcr': [ORIGINAL_QCR_REPO, ORIGINAL_QCR_COMMIT, 'data/']}
    elif dataset == 'slake':
        # lib/utils/run.sh, SLAKE part. GloVe 6B-300d and the pretrained auto-encoder come from the
        # MEVF bundle on HF (the same auto-encoder file as the RAD bundle; QCR ships no SLAKE one).
        mevf = fetch('mevf_rad', cache_dir)
        glove = data_root / 'glove.6B' / 'glove.6B.300d.txt'
        if not glove.exists():
            _extract_member(mevf, 'glove/glove.6B.300d.txt', glove)
        slake_dir.mkdir(parents=True, exist_ok=True)
        _extract_member(mevf, 'pretrained_ae.pth', slake_dir / 'pretrained_ae.pth')
        if sha256(slake_dir / 'pretrained_ae.pth') != RAD_SHA256['pretrained_ae.pth']:
            raise ValueError('Unexpected auto-encoder checkpoint')
        for key, name in (('slake_train', 'train.json'), ('slake_test', 'test.json')):
            shutil.copyfile(fetch(key, cache_dir), slake_dir / name)
        if not (slake_dir / 'imgs').exists():
            safe_extract(fetch('slake_imgs', cache_dir), slake_dir)
        counts = _check_counts('slake', slake_dir)
        run([py, 'create_dictionary.py', slake_dir.as_posix()], cwd=utils_dir)
        run([py, 'create_label.py', slake_dir.as_posix() + '/'], cwd=utils_dir)
        run([py, 'create_img2idx.py', slake_dir / 'train.json', slake_dir / 'test.json', slake_dir / 'imgid2idx.json'], cwd=utils_dir)
        for size, channels in ((128, 1), (250, 3)):
            run([py, 'create_resized_images.py', slake_dir / 'imgid2idx.json', slake_dir / 'imgs',
                 size, slake_dir / f'images{size}x{size}.pkl', channels], cwd=utils_dir)
        assets = {k: list(ASSETS[k][:4]) for k in ('mevf_rad', 'slake_imgs', 'slake_train', 'slake_test')}
    else:
        raise ValueError('dataset must be rad or slake')
    info = {'dataset': dataset, 'counts': counts, 'assets': assets}
    marker.write_text(json.dumps(info, indent=2), encoding='utf-8')
    return info


def make_config(qcr, dataset, clip_path, run_index, n_epoch=None):
    """Copy the authors' config; only CLIP_PATH, the run NAME and (quick checks only) N_EPOCH change."""
    qcr = Path(qcr)
    spec = RUNS[dataset]
    text = (qcr / 'configs' / spec['config']).read_text(encoding='utf-8')
    text, n = re.subn(r'CLIP_PATH: ".*"', f'CLIP_PATH: "{Path(clip_path).as_posix()}"', text)
    assert n == 1
    suffix = f'.run{run_index}' + (f'.epochs{n_epoch}' if n_epoch else '')
    text, n = re.subn(r"^NAME: '(.*)'", lambda m: f"NAME: '{m.group(1)}{suffix}'", text, flags=re.M)
    assert n == 1
    if n_epoch:
        text, n = re.subn(r'N_EPOCH: \d+', f'N_EPOCH: {int(n_epoch)}', text)
        assert n == 1
    out = qcr / 'configs' / f'repro_{dataset}{suffix}.yaml'
    out.write_text(text, encoding='utf-8')
    return out


def config_value(config_path, key):
    match = re.search(rf'^\s*{key}:\s*[\'"]?([^\'"\n#]+)', Path(config_path).read_text(encoding='utf-8'), re.M)
    return match.group(1).strip()


VAL_RE = re.compile(r'\[Validate\] Val_Acc:([\d.]+)%\s*\|\s*Open_ACC:([\d.]+)%\s*\|\s*Close_ACC:([\d.]+)%')


def parse_log(log_path):
    """Per-epoch test accuracy from medVQA.log; best epoch selected exactly as train.py does."""
    epochs = [{'epoch': i, 'overall': float(a), 'open': float(o), 'closed': float(c)}
              for i, (a, o, c) in enumerate(VAL_RE.findall(Path(log_path).read_text(encoding='utf-8', errors='replace')))]
    best = None
    for row in epochs:  # train.py keeps the first epoch with a strictly higher overall accuracy
        if row['overall'] > (best['overall'] if best else 0):
            best = row
    return {'epochs_logged': len(epochs), 'best': best, 'last': epochs[-1] if epochs else None}


def train(qcr, dataset, clip_path, run_index=0, n_epoch=None, anomaly=False, keep_all_best=False):
    """Run the authors' main/main.py with their config (see make_config) and parse its log."""
    import os
    qcr = Path(qcr)
    config = make_config(qcr, dataset, clip_path, run_index, n_epoch)
    env = dict(os.environ, QCR_DETECT_ANOMALY='1' if anomaly else '0', QCR_KEEP_ALL_BEST='1' if keep_all_best else '0')
    run([sys.executable, 'main/main.py', '--cfg', config.relative_to(qcr).as_posix()], cwd=qcr, env=env)
    out = qcr / config_value(config, 'OUTPUT_DIR') / config_value(config, 'NAME')
    result = parse_log(out / 'medVQA.log')
    expected_epochs = int(config_value(config, 'N_EPOCH'))
    result.update(dataset=dataset, encoder=RUNS[dataset]['encoder'], run=run_index, config=config.name,
                  full_protocol=n_epoch is None and result['epochs_logged'] == expected_epochs,
                  paper_open_closed_overall=RUNS[dataset]['paper'], output_dir=str(out))
    (out / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def summarize(results):
    """Mean/std over runs of each run's best-epoch test accuracy (the authors' protocol) and last epoch."""
    import statistics
    table = {}
    for dataset in RUNS:
        rows = [r for r in results if r['dataset'] == dataset and r['best']]
        if not rows:
            continue
        entry = {'runs': len(rows), 'full_protocol': all(r['full_protocol'] for r in rows),
                 'paper_open_closed_overall': RUNS[dataset]['paper']}
        for key in ('open', 'closed', 'overall'):
            values = [r['best'][key] for r in rows]
            entry[key] = statistics.mean(values)
            entry[key + '_std'] = statistics.stdev(values) if len(values) > 1 else 0.0
            entry['last_epoch_' + key] = statistics.mean(r['last'][key] for r in rows)
        table[dataset] = entry
    return table
