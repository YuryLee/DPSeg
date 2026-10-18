#!/usr/bin/env python3
"""Check selected checkpoint reload against saved validation predictions only."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rgbd_seg.course_optimized import file_sha, load_for_inference


def compare_predictions(expected, actual, *, score_tolerance=1e-6):
    """Require identical ordered instance masks; allow only FP32 score roundoff."""
    if len(expected) != len(actual):
        raise ValueError('Reload changed the number of predictions')
    maximum = 0.0
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left['category_id'] != right['category_id'] or left['segmentation'] != right['segmentation']:
            raise ValueError(f'Reload changed mask/category/order at instance {index}')
        scores = (float(left['score']), float(right['score']))
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in scores):
            raise ValueError('Invalid prediction score')
        difference = abs(scores[0] - scores[1])
        if difference > score_tolerance:
            raise ValueError(f'Reload changed confidence at instance {index}: {difference}')
        maximum = max(maximum, difference)
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--images', type=int, default=2)
    args = parser.parse_args()
    if args.images < 1:
        parser.error('--images must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)

    import torch
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_depth_data import load_inference_sample

    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run = args.run.resolve()
    state = json.loads((run/'run.json').read_text())
    if state['status'] != 'completed' or state['test_used']:
        raise ValueError('Require completed validation-only run')
    protocol = json.loads(args.protocol.read_text())
    # Hashes tie the original predictions and the reloaded model to one epoch.
    model, reload_receipt = load_for_inference(run/'best.pt', args.protocol, device=args.device)
    if file_sha(run/'best.pt') != state['best_checkpoint_sha256']:
        raise ValueError('Best checkpoint no longer matches completed run')
    selection = json.loads((run/'selection.json').read_text())
    entry = next(x for x in selection['epochs'] if x['epoch'] == selection['best_epoch'])
    if (entry['epoch'] != reload_receipt['epoch']
            or entry['model_sha256'] != reload_receipt['model_sha256']):
        raise ValueError('Validation selection differs from reloaded model')
    evaluation = Path(entry['output']).resolve()
    if not evaluation.is_relative_to(run):
        raise ValueError('Validation output is outside this run')
    receipt = json.loads((evaluation/'evaluation_run.json').read_text())
    if receipt['status'] != 'completed' or receipt['split'] != 'val':
        raise ValueError('Require completed validation predictions')
    predictions = evaluation/'predictions.jsonl'
    if file_sha(predictions) != receipt['predictions_sha256']:
        raise ValueError('Saved validation predictions changed')
    # First two rank-0 images are deterministic and use the same device class
    # when invoked on the original rank-0 GPU. No ground-truth selection.
    rank_ids = json.loads((evaluation/'rank0.json').read_text())['ordered_ids']
    if args.images > len(rank_ids):
        raise ValueError('Requested more images than rank 0 evaluated')
    ids = rank_ids[:args.images]
    references = {}
    with predictions.open() as stream:
        for line in stream:
            record = json.loads(line)
            if record['image_id'] in ids:
                if record['image_id'] in references:
                    raise ValueError('Duplicate reference image')
                references[record['image_id']] = record['predictions']
    if set(references) != set(ids):
        raise ValueError('Missing reference images')
    checks = []
    with CourseArchive(protocol['archive'], split='val') as dataset, torch.inference_mode():
        first = load_inference_sample(dataset, ids[0])
        for _ in range(3):
            model.predict_sample(first, score_threshold=.01, max_detections=100)
        for image_id in ids:
            sample = load_inference_sample(dataset, image_id)
            actual = model.predict_sample(sample, score_threshold=.01, max_detections=100)
            difference = compare_predictions(references[image_id], actual)
            checks.append({'image_id': image_id, 'instances': len(actual),
                           'exact_masks_categories_order': True, 'max_score_difference': difference})
    result = {'status': 'passed', 'scope': 'selected_checkpoint_validation_prediction_reload',
        'limitations': 'Bounded engineering check, not a new full validation evaluation',
        'reload': reload_receipt, 'reference_predictions_sha256': receipt['predictions_sha256'],
        'score_absolute_tolerance': 1e-6, 'checks': checks, 'test_used': False,
        'device': args.device, 'checker_sha256': file_sha(__file__)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'passed', 'images': len(checks), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
