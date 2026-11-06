"""Fast checks for biomedclip_meter (no GPU, no downloads). Run: python tests/test_biomedclip_meter.py
The CPU training smoke test is tests/biomedclip_cpu_smoke.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import biomedclip_meter as b  # noqa: E402


def test_normalize_answer_matches_vqa_rules():
    assert b.normalize_answer('Yes') == 'yes'
    assert b.normalize_answer('The left lung.') == 'left lung'
    assert b.normalize_answer('two') == '2'
    assert b.normalize_answer('X-ray') == 'x ray'
    assert b.normalize_answer('1,000') == '1000'
    assert b.normalize_answer('dont know') == "don't know"


def test_history_selection():
    hist = [{'epoch': 0, 'test': {'open': 10, 'closed': 50, 'overall': 30}, 'validation': {'overall': 40}},
            {'epoch': 1, 'test': {'open': 20, 'closed': 60, 'overall': 40}, 'validation': {'overall': 39}},
            {'epoch': 2, 'test': {'open': 25, 'closed': 55, 'overall': 40}, 'validation': {'overall': 40}}]
    r = b.summarize_history(hist)
    assert r['best_test_epoch']['epoch'] == 1          # first strictly-best test epoch (QCR/M3AE)
    assert r['last_epoch']['open'] == 25
    assert r['validation_selected']['epoch'] == 0      # earliest epoch with the best validation accuracy
    rows = [dict(r, dataset='slake', full_protocol=True)] * 2
    s = b.summarize(rows)['slake']
    assert s['runs'] == 2 and s['best_test_epoch']['overall'] == 40 and s['best_test_epoch']['overall_std'] == 0


def _fake_run(work, dataset='rad', overrides=None, epochs=3):
    out = b.run_dir(work, dataset, 0, overrides)
    out.mkdir(parents=True)
    hist = [{'epoch': e, 'train_loss': 1.0, 'train_acc': 50.0 + e, 'test': {'open': e, 'closed': e, 'overall': [30, 40, 35][e]}}
            for e in range(epochs)]
    b._write_json(out / 'history.json', hist)
    for e in range(epochs):
        (out / f'test_predictions_epoch{e}.json').write_text('[]', encoding='utf-8')
    return out


def test_run_dir_matches_fit_tags():
    assert b.run_dir('w', 'rad').as_posix() == 'w/runs/biomedclip_meter_rad_run0/full'
    assert b.run_dir('w', 'slake', 1, {'epochs': 2, 'image_size': 448}).name == 'quick_epochs2_image_size448'
    assert b.run_dir('w', 'rad', limit=(4, 4)).name == 'quick_'


def test_progress_and_status():
    import tempfile
    with tempfile.TemporaryDirectory() as work:
        _fake_run(work)
        jobs = [{'dataset': 'rad', 'run': 0, 'overrides': {}}, {'dataset': 'slake', 'run': 0, 'overrides': {'epochs': 2}}]
        rad, slake = b.progress(work, jobs)
        assert rad['epochs'] == '3/50' and rad['best_epoch'] == 1 and rad['best_overall'] == 40 and rad['last_train_acc'] == 52
        assert slake['epochs'] == '0/2' and slake['best_overall'] is None
        assert b.trainer_status(work)['state'] == 'not started'
        log = Path(work) / 'train.log'
        log.write_text(f'{b.START_MARK} a ===\n{b.DONE_MARK}\n{b.START_MARK} b ===\nTraceback: boom\n', encoding='utf-8')
        status = b.trainer_status(work)
        assert status['state'] == 'failed' and 'boom' in status['tail']   # DONE of an older launch does not count
        log.write_text(log.read_text(encoding='utf-8') + b.DONE_MARK + '\n', encoding='utf-8')
        assert b.trainer_status(work)['state'] == 'finished'


def test_backup_roundtrip_with_extractor():
    import json
    import subprocess
    import tempfile
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as work:
        out = _fake_run(work)
        (Path(work) / 'results.json').write_text('[]', encoding='utf-8')
        text = b.backup_text(work)
        nb = Path(work) / 'nb.ipynb'
        nb.write_text(json.dumps({'cells': [{'cell_type': 'code', 'source': [], 'outputs': [
            {'output_type': 'stream', 'name': 'stdout', 'text': ['0.01 MB of text\n', text + '\n']}]}]}), encoding='utf-8')
        dest = Path(work) / 'restored'
        subprocess.run([sys.executable, str(root / 'extract_notebook_backup.py'), str(nb), str(dest)], check=True, capture_output=True)
        rel = out.relative_to(work).as_posix()
        assert (dest / 'results.json').exists() and (dest / rel / 'history.json').exists()
        # best (1) and last (2) epoch predictions only
        assert sorted(p.name for p in (dest / rel).glob('test_predictions_*')) == ['test_predictions_epoch1.json', 'test_predictions_epoch2.json']


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
