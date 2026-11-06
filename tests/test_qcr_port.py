"""Fast checks for qcr_port (no GPU, no downloads). Run: python tests/test_qcr_port.py

Patch checks need the official code: upstream/PubMedCLIP at QCR_COMMIT (git clone + checkout).
The full CPU training smoke test is tests/qcr_cpu_smoke.py.
"""
import ast
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import qcr_port  # noqa: E402

UPSTREAM = ROOT / 'upstream' / 'PubMedCLIP'


def test_patches_apply_once_and_are_idempotent():
    if not UPSTREAM.exists():
        print('skip: upstream/PubMedCLIP not present')
        return
    with tempfile.TemporaryDirectory() as tmp:
        qcr, applied = qcr_port.setup_code(tmp, source=UPSTREAM)
        assert len(applied) == len(qcr_port.PATCHES)
        qcr_port.setup_code(tmp, source=UPSTREAM)  # second application must be a no-op
        for rel in {p[0] for p in qcr_port.PATCHES}:
            ast.parse((qcr / rel).read_text(encoding='utf-8'))
        utils = (qcr / 'lib/utils/utils.py').read_text(encoding='utf-8')
        assert 'torch._six' not in utils and 'pickle_module=_tfidf_pickle' in utils
        assert 'inplace=False' in (qcr / 'main/classifier.py').read_text(encoding='utf-8')


def test_config_only_changes_clip_path_name_and_quick_epochs():
    if not UPSTREAM.exists():
        print('skip: upstream/PubMedCLIP not present')
        return
    with tempfile.TemporaryDirectory() as tmp:
        qcr = Path(tmp) / 'QCR_PubMedCLIP'
        shutil.copytree(UPSTREAM / 'QCR_PubMedCLIP' / 'configs', qcr / 'configs')
        for dataset, spec in qcr_port.RUNS.items():
            original = (qcr / 'configs' / spec['config']).read_text(encoding='utf-8').splitlines()
            full = qcr_port.make_config(qcr, dataset, '/w/model.pth', 2).read_text(encoding='utf-8').splitlines()
            changed = [(a, b) for a, b in zip(original, full) if a != b]
            assert len(original) == len(full) and len(changed) == 2, changed
            assert qcr_port.config_value(qcr / 'configs' / f'repro_{dataset}.run2.yaml', 'N_EPOCH') == '200'
            quick = qcr_port.make_config(qcr, dataset, '/w/model.pth', 0, n_epoch=3)
            assert qcr_port.config_value(quick, 'N_EPOCH') == '3'


def test_parse_log_selects_first_strict_best_like_train_py():
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / 'medVQA.log'
        line = '[Validate] Val_Acc:{:.6f}%  |  Open_ACC:{:.6f}%   |  Close_ACC:{:.6f}%\n'
        log.write_text(line.format(50, 40, 60) + line.format(70, 55, 80) + line.format(70, 60, 76), encoding='utf-8')
        result = qcr_port.parse_log(log)
        assert result['epochs_logged'] == 3
        assert result['best'] == {'epoch': 1, 'overall': 70.0, 'open': 55.0, 'closed': 80.0}
        assert result['last']['open'] == 60.0
        rows = [dict(result, dataset='rad', full_protocol=True)] * 2
        summary = qcr_port.summarize(rows)['rad']
        assert summary['runs'] == 2 and summary['overall'] == 70.0 and summary['overall_std'] == 0.0


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
