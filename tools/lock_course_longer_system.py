#!/usr/bin/env python3
"""Lock one validation-only choice across both longer-training model families."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FAMILIES = {'selected_base', 'original_base_extended_heads'}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bind(path, artifacts):
    path = Path(path).resolve()
    digest = sha(path)
    require(str(path) not in artifacts or artifacts[str(path)] == digest, 'Artifact changed during selection')
    artifacts[str(path)] = digest
    return digest


def rank_all(families, historical):
    require(set(families) == FAMILIES, 'Both planned model families must be accounted for')
    rows = []
    for family, variants in [('historical', historical), *sorted(families.items())]:
        for variant, metrics in variants.items():
            values = {key: metrics[key] for key in ('mask_ap_small', 'mask_ap')}
            require(all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
                        for v in values.values()), 'Invalid validation AP')
            rows.append({'family': family, 'variant': variant, **values})
    require(bool(historical) and all(families.values()), 'Empty family cannot enter final selection')
    # Family name is only a deterministic tie-break after APs, AP and historical retention.
    return sorted(rows, key=lambda r: (-r['mask_ap_small'], -r['mask_ap'],
                                      r['family'] != 'historical', r['family'], r['variant']))


def inspect_evaluation(folder, expected_ids, metrics, artifacts):
    evaluation = read(folder / 'evaluation.json')
    require(evaluation['status'] == 'completed' and evaluation['split'] == 'val'
            and evaluation['images'] == len(expected_ids) == 2584, 'Incomplete full validation')
    require(evaluation['metrics'] == metrics, 'Summary differs from actual evaluator output')
    require(evaluation['iou_type'] == 'segm' and evaluation['max_detections'] == [1, 10, 100]
            and evaluation['area_ranges'][1] == [0, 1024]
            and evaluation['ground_truth_area'] == 'original annotation area preserved'
            and len(evaluation['iou_thresholds']) == 10
            and all(math.isclose(v, .5 + .05 * i) for i, v in enumerate(evaluation['iou_thresholds'])),
            'Evaluation does not use the agreed COCO mask APs definition')
    prediction = folder / 'predictions.jsonl'
    require(bind(prediction, artifacts) == evaluation['predictions_sha256'], 'Scored predictions changed')
    ids, count = [], 0
    with prediction.open() as stream:
        for line in stream:
            row = json.loads(line)
            ids.append(row['image_id'])
            count += len(row['predictions'])
            require(len(row['predictions']) <= 100, 'Too many instances')
    require(ids == expected_ids and count == evaluation['predicted_instances'], 'Prediction coverage differs')
    bind(folder / 'evaluation.json', artifacts)


def inspect_family(item, historical_path, ids, artifacts):
    from tools.run_course_longer_tiled import ALL_ARMS, rank_candidates, verify
    protocol_path, output = Path(item['protocol']), Path(item['output'])
    protocol = read(protocol_path)
    verify(protocol)
    digest = bind(protocol_path, artifacts)
    pp = read(protocol['point_protocol']['path'])
    point_report = read(protocol['point_validation']['path'])
    expected_schema = ('course_point_continuation_v1' if item['name'] == 'original_base_extended_heads'
                       else 'course_longer_point_v1')
    require(pp['schema'] == expected_schema and pp['scope'] == 'full_experiment'
            and pp['evaluation_policy'] == 'validation_only'
            and point_report['status'] == 'completed' and point_report['scope'] == 'full_experiment'
            and not point_report['test_used'] and len(point_report['validation_history']) == pp['epochs']
            and point_report['protocol_sha256'] == protocol['point_protocol']['sha256'],
            'Family does not contain the required completed point-training branch')
    audit = read(item['point_audit'])
    require(audit['status'] == 'passed' and audit['scope'] == 'full_experiment'
            and audit['artifact_sha256'].get(str(Path(protocol['point_protocol']['path']).resolve()))
                == protocol['point_protocol']['sha256'] and not audit['test_used'],
            'Point training has not passed its full audit')
    bind(item['point_audit'], artifacts)
    for path, expected_digest in audit['artifact_sha256'].items():
        require(bind(path, artifacts) == expected_digest, 'Audited point artifact changed')
    for name in ('weights', 'training_protocol'):
        require(bind(pp[name]['path'], artifacts) == pp[name]['sha256'], 'Selected base identity changed')
    if item['name'] == 'selected_base':
        decision = read(pp['base_selection']['path'])
        require(bind(pp['base_selection']['path'], artifacts) == pp['base_selection']['sha256']
                and decision['status'] == 'selected_after_convergence_review'
                and decision['selection_split'] == 'val' and not decision['test_used']
                and decision['selected']['checkpoint_sha256'] == pp['weights']['sha256'],
                'Selected base is not the locked convergence-review result')
    require(Path(protocol['historical_comparison']['path']).resolve() == historical_path.resolve(),
            'Families use different historical comparisons')
    require(not (output / 'test_run.json').exists() and not (output / 'test').exists(),
            'Global selection must precede test inference in every new family')
    run, choice, metrics = (read(output / n) for n in ('val_run.json', 'selection.json', 'val/metrics.json'))
    require(run['status'] == 'completed' and run['phase'] == 'val' and run['protocol_sha256'] == digest,
            'Family validation is incomplete or uses another protocol')
    require(choice['status'] == 'locked' and choice['selection_split'] == 'val'
            and choice['selected_before_test'] and not choice['test_used']
            and choice['protocol_sha256'] == digest
            and choice['val_metrics_sha256'] == sha(output / 'val/metrics.json'), 'Invalid family selection')
    require(set(metrics) == set(ALL_ARMS), 'Missing whole-image or tiled variants')
    selected_head = point_report['selection']['selected']
    require(selected_head['checkpoint']['sha256'] == protocol['head']['sha256']
            and protocol['head_kind'] == ('depth' if selected_head['variant'].startswith('point_depth') else 'plain'),
            'System does not use the point head selected on validation')
    expected = rank_candidates(metrics, read(historical_path)['val'])
    require(choice['candidates'] == expected and choice['selected'] == expected[0], 'Family ranking differs')
    require(read(output / 'val/image_ids.json') == ids, 'Family did not use the official validation split')
    require(read(output / 'smoke/engineering.json')['status'] == 'passed', 'Engineering check did not pass')
    for variant, values in metrics.items():
        inspect_evaluation(output / 'val' / variant, ids, values, artifacts)
    for name in ('val_run.json', 'selection.json', 'val/metrics.json', 'val/image_ids.json', 'smoke/engineering.json'):
        bind(output / name, artifacts)
    for name in ('head', 'point_protocol', 'point_validation'):
        require(bind(protocol[name]['path'], artifacts) == protocol[name]['sha256'], 'Family input changed')
    for name, digest in protocol['source_sha256'].items():
        require(bind(ROOT / name, artifacts) == digest, 'Frozen implementation changed')
    return metrics


def inspect_retained_parent(item, historical_protocol, historical_metrics, artifacts):
    """An audited continuation may select the exact original base/head again."""
    from tools.run_course_point_continuation import parent_inputs, select_validation, selection_history
    from tools.run_course_point_refinement import checked
    protocol = checked(item['point_protocol'])
    report = read(item['point_report'])
    audit = read(item['point_audit'])
    require(report['status'] == 'completed' and report['scope'] == 'full_experiment'
            and not report['test_used'] and report['protocol_sha256'] == protocol['_sha256']
            and len(report['validation_history']) == protocol['epochs'], 'Point continuation is incomplete')
    require(audit['status'] == 'passed' and audit['scope'] == 'full_experiment'
            and audit['protocol_sha256'] == protocol['_sha256'] and not audit['test_used'],
            'Retained parent requires the completed continuation audit')
    parent_inputs(protocol)
    choice = select_validation(selection_history(protocol, report['validation_history']))
    require(choice == report['selection'], 'Point-head selection differs')
    selected = choice['selected']
    require(selected['checkpoint']['sha256'] == historical_protocol['head']['sha256']
            and selected['variant'].startswith('point_depth'), 'Selected head is not the historical depth head')
    old_point = read(historical_protocol['point_protocol']['path'])
    require(protocol['weights']['sha256'] == old_point['weights']['sha256']
            and protocol['point_policy'] == old_point['point_policy'], 'Retained base or point policy differs')
    for name in ('point_protocol', 'point_report', 'point_audit'):
        bind(item[name], artifacts)
    for path, digest in audit['artifact_sha256'].items():
        require(bind(path, artifacts) == digest, 'Audited point artifact changed')
    # These are exact duplicate system candidates, not an assertion that all new epochs tie.
    return historical_metrics


def lock(manifest_path):
    from rgbd_seg.course_archive import CourseArchive
    manifest, artifacts = read(manifest_path), {}
    items = manifest['families']
    require(len(items) == len(FAMILIES) and {v['name'] for v in items} == FAMILIES,
            'Both planned model families are required exactly once')
    history_path = Path(manifest['historical_comparison'])
    history, hp = read(history_path), read(manifest['historical_protocol'])
    require(history['status'] == 'completed' and history['protocol_sha256'] == sha(manifest['historical_protocol']),
            'Historical comparison provenance differs')
    with CourseArchive(hp['archive'], 'val') as data:
        ids = data.image_ids
    require(len(ids) == 2584, 'Official validation split differs')
    for variant, metrics in history['val'].items():
        inspect_evaluation(history_path.parent / 'val' / variant, ids, metrics, artifacts)
    families = {}
    for item in items:
        if item.get('mode') == 'retained_historical_parent':
            require(item['name'] == 'original_base_extended_heads', 'Only original-base continuation can retain this parent')
            families[item['name']] = inspect_retained_parent(item, hp, history['val'], artifacts)
        else:
            require(item.get('mode', 'evaluated') == 'evaluated', 'Unknown family mode')
            require(read(item['protocol'])['archive'] == hp['archive'], 'Different dataset archives')
            families[item['name']] = inspect_family(item, history_path, ids, artifacts)
    for path in (manifest_path, history_path, manifest['historical_protocol'], Path(__file__)):
        bind(path, artifacts)
    ranked = rank_all(families, history['val'])
    return {'status': 'locked', 'scope': 'both_families_full_validation', 'selected': ranked[0],
            'candidates': ranked, 'selection_split': 'val', 'test_used_for_selection': False,
            'selected_before_new_test': True, 'validation_images': len(ids),
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'families': items, 'artifact_sha256': artifacts,
            'limitations': ['Historical test scores already existed; only validation metrics enter ranking',
                           'One seed; best among evaluated candidates',
                           'Head epoch/type selected on whole-image validation before tiled comparison']}


def verify_lock(path):
    result = read(path)
    require(result['status'] == 'locked' and result['scope'] == 'both_families_full_validation'
            and result['selection_split'] == 'val' and not result['test_used_for_selection']
            and result['selected_before_new_test'], 'Not a complete global validation decision')
    grouped = {name: {} for name in (*sorted(FAMILIES), 'historical')}
    for candidate in result['candidates']:
        group = grouped[candidate['family']]
        require(candidate['variant'] not in group, 'Duplicate locked candidate')
        group[candidate['variant']] = candidate
    ranked = rank_all({name: grouped[name] for name in FAMILIES}, grouped['historical'])
    require(result['candidates'] == ranked and result['selected'] == ranked[0], 'Locked ranking changed')
    for artifact, digest in result['artifact_sha256'].items():
        require(sha(artifact) == digest, f'Locked selection input changed: {artifact}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    create = commands.add_parser('lock')
    create.add_argument('--manifest', type=Path, required=True)
    create.add_argument('--output', type=Path, required=True)
    test = commands.add_parser('test')
    test.add_argument('--selection', type=Path, required=True)
    test.add_argument('--family', choices=sorted(FAMILIES), required=True)
    args = parser.parse_args()
    if args.action == 'lock':
        require(not args.output.exists(), 'Selection output already exists')
        result = lock(args.manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        print(json.dumps(result['selected']))
    else:
        selection = verify_lock(args.selection)
        selection_digest = sha(args.selection)
        family = next(v for v in selection['families'] if v['name'] == args.family)
        require(family.get('mode', 'evaluated') == 'evaluated', 'Retained historical family already has its test outputs')
        subprocess.run([sys.executable, str(ROOT / 'tools/run_course_longer_tiled.py'),
                        '--protocol', family['protocol'], '--output', family['output'], '--phase', 'test'],
                       cwd=ROOT, check=True)
        require(sha(args.selection) == selection_digest, 'Global decision changed during test')
        verify_lock(args.selection)


if __name__ == '__main__':
    main()
