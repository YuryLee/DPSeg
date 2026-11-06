#!/usr/bin/env python3
"""Rescore the frozen reference with current annotations before SAHI reporting."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluate_course_swinl_ablation import read, digest, write
from tools.evaluate_course_sahi_merge import SEMANTICS
from tools.summarize_course_swinl_removal import check_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'reports/course_sahi_merge_v1')
    args = parser.parse_args()
    run = args.run.resolve()
    plan = read(run/'protocol.json')
    reference = plan['full_reference']
    for key in ('predictions', 'evaluation'):
        if digest(reference[key]) != reference[f'{key}_sha256']:
            raise ValueError('Frozen reference changed')
    output = run/'reference_rescore'
    output.mkdir(exist_ok=True)
    evaluation_path = output/'evaluation.json'
    if evaluation_path.exists():
        result = read(evaluation_path)
    else:
        from rgbd_seg.course_archive import CourseArchive
        from rgbd_seg.course_dafusion_eval import score_predictions
        with CourseArchive(plan['archive'], split='test') as data:
            if data.image_ids != plan['image_ids']:
                raise ValueError('Test IDs changed')
            result = score_predictions(data, Path(reference['predictions']), output)
    check_sources(result['source_sha256'])
    original = read(reference['evaluation'])
    if (result['status'] != 'completed' or result['images'] != 2112 or result['split'] != 'test'
            or result['metrics'] != reference['metrics']
            or result['predictions_sha256'] != reference['predictions_sha256']
            or {k: result[k] for k in SEMANTICS} != {k: original[k] for k in SEMANTICS}):
        raise ValueError('Fresh reference scoring differs from the historical baseline')
    receipt = dict(status='passed', all_metrics_exact=True, images=2112,
        reference_predictions_sha256=reference['predictions_sha256'],
        evaluation=str(evaluation_path), evaluation_sha256=digest(evaluation_path),
        metrics=result['metrics'], checker_sha256=digest(Path(__file__)))
    write(output/'verification.json', receipt)
    print('Reference verified: all 12 COCO metrics exactly reproduce the historical baseline.')


if __name__ == '__main__':
    main()
