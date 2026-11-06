"""CPU smoke test of biomedclip_meter.fit on a tiny subset (not a benchmark).

Usage: python tests/biomedclip_cpu_smoke.py <work_dir> <cache_dir> <rad|slake> [image_size]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import biomedclip_meter  # noqa: E402

work, cache, dataset = sys.argv[1:4]
size = int(sys.argv[4]) if len(sys.argv) > 4 else 224
result = biomedclip_meter.fit(dataset, work, cache, overrides=dict(epochs=1, batch_size=8, micro_batch=4, num_workers=0,
                              image_size=size, amp=False), device='cpu', limit=(16, 8))
print({k: result[k] for k in ('best_test_epoch', 'last_epoch', 'num_labels', 'counts', 'full_protocol', 'truncated_questions')},
      result.get('validation_selected'))
assert result['epochs_logged'] == 1 and not result['full_protocol']
