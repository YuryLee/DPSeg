#!/usr/bin/env python3
"""Audit completed full validation and apply the frozen refinement selection gate."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pycocotools import mask as mask_utils
from rgbd_seg.course_archive import CourseArchive
from tools.eval_course_depth_refinement import checked_protocol
from tools.select_course_resolution import read, sha, require


def area(prediction):
    rle = prediction['segmentation']
    return int(mask_utils.area({'size': rle['size'], 'counts': rle['counts'].encode('ascii')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Decision output must be new')
    p = checked_protocol(args.protocol)
    require(p['scope'] == 'full_validation', 'Smoke is not selection evidence')
    run, terminal = read(args.run/'run.json'), read(args.run/'exit.json')
    require(run['status'] == terminal['status'] == 'completed'
            and run['exit_code'] == terminal['exit_code'] == 0
            and terminal['worker_exit_codes'] == [0]*7, 'Incomplete run')
    require(run['protocol_sha256'] == terminal['protocol_sha256'] == sha(args.protocol)
            and run['source_sha256'] == p['source_sha256'], 'Runtime identity differs')
    require(run['images'] == 2584 and run['world_size'] == 7
            and not run['test_accessed'] and not run['ground_truth_in_refinement'], 'Wrong run scope')
    with CourseArchive(p['archive'], 'val') as dataset:
        ids = list(dataset.image_ids)
    require(read(args.run/'image_ids.json') == ids, 'Validation IDs differ')
    baseline_decision = read(p['decision']['path'])
    baseline = baseline_decision['baseline']
    original = [json.loads(x) for x in Path(p['baseline_predictions']['path']).read_text().splitlines()]
    require([x['image_id'] for x in original] == ids, 'Baseline coverage differs')
    audits, eligible = {}, []
    gate = p['adoption_gate']
    for mode in p['modes']:
        folder = args.run/mode
        evaluation, receipt = read(folder/'evaluation.json'), read(folder/'evaluation_run.json')
        require(evaluation['status'] == receipt['status'] == 'completed'
                and evaluation['scope'] == receipt['scope'] == 'full_split'
                and evaluation['split'] == receipt['split'] == 'val'
                and evaluation['images'] == receipt['images'] == 2584, 'Incomplete scoring')
        require(evaluation['metrics'] == run['metrics'][mode], 'Reported metrics differ')
        require(sha(folder/'predictions.jsonl') == receipt['predictions_sha256'] == evaluation['predictions_sha256']
                and sha(folder/'evaluation.json') == receipt['evaluation_sha256'], 'Output hash differs')
        merged = [json.loads(x) for x in (folder/'predictions.jsonl').read_text().splitlines()]
        require([x['image_id'] for x in merged] == ids, 'Merged IDs differ')
        for rank in range(7):
            shard = read(args.run/f'rank{rank}.json')
            path = folder/f'predictions_rank{rank}.jsonl'
            require(shard['status'] == 'completed' and shard['physical_gpu'] == p['gpus'][rank]
                    and shard['ordered_ids'] == ids[rank::7], 'Shard coverage differs')
            require(sha(path) == shard['predictions_sha256'][mode]
                    and sha(args.run/f'source_rank{rank}.jsonl') == shard['source_sha256'], 'Shard hash differs')
            require([json.loads(x) for x in path.read_text().splitlines()] == merged[rank::7],
                    'Merge differs from shard')
        count = changed = net_area = 0
        for old, new in zip(original, merged):
            require(old['file_name'] == new['file_name'], 'Image identity differs')
            a, b = old['predictions'], new['predictions']
            require([(x['category_id'], x['score']) for x in a] ==
                    [(x['category_id'], x['score']) for x in b], 'Scores/order/count differ')
            require(len(b) <= 100, 'Prediction cap exceeded')
            count += len(b)
            for x, y in zip(a, b):
                require(y['segmentation']['size'] == [1024,1024] and area(y) > 0, 'Invalid refined mask')
                changed += x['segmentation'] != y['segmentation']
                net_area += area(y)-area(x)
        stats = run['refinement_stats'][mode]
        require(count == stats['instances'] == receipt['predicted_instances'] == evaluation['predicted_instances']
                and changed == stats['changed_instances']
                and net_area == stats['added_pixels']-stats['removed_pixels'], 'Refinement statistics differ')
        metrics = evaluation['metrics']
        gain = metrics['mask_ap_small']-baseline['metrics']['mask_ap_small']
        drop = baseline['metrics']['mask_ap']-metrics['mask_ap']
        adopt = gain >= gate['APsmall_gain_at_least'] and drop <= gate['AP_drop_at_most']
        if adopt:
            eligible.append(mode)
        audits[mode] = {'status':'passed', 'images':2584, 'instances':count,
            'scores_order_count_preserved':True, 'changed_instances':changed, 'net_area_change':net_area,
            'metrics':metrics, 'APsmall_gain':gain, 'AP_drop':drop, 'eligible':adopt,
            'predictions':{'path':str((folder/'predictions.jsonl').resolve()),'sha256':receipt['predictions_sha256']},
            'evaluation':{'path':str((folder/'evaluation.json').resolve()),'sha256':receipt['evaluation_sha256']}}
    chosen = max(eligible, key=lambda m:(audits[m]['metrics']['mask_ap_small'],
                 audits[m]['metrics']['mask_ap'], m == 'spatial')) if eligible else 'native'
    delta = audits['depth']['metrics']['mask_ap_small']-audits['spatial']['metrics']['mask_ap_small']
    result = {'status':'completed', 'finished_at_utc':datetime.now(timezone.utc).isoformat(),
        'material_passport':{'origin_skill':'academic-research-suite/experiment-agent','mode':'run',
            'scope':'Full validation audit and predeclared selection','verification_status':'VERIFIED_OUTPUT_BINDINGS'},
        'selected_method':chosen, 'base_model_method':'native', 'resolution':1024,
        'baseline':baseline, 'baseline_protocol':p['baseline_protocol'], 'adoption_gate':gate,
        'variants':audits, 'depth_minus_spatial_APsmall':delta,
        'selected_metrics':baseline['metrics'] if chosen == 'native' else audits[chosen]['metrics'],
        'test_used':False, 'protocol':{'path':str(args.protocol.resolve()),'sha256':sha(args.protocol)},
        'run':{'path':str((args.run/'run.json').resolve()),'sha256':sha(args.run/'run.json')},
        'script_sha256':sha(__file__), 'limits':['One fixed configuration; no repeated-seed significance test',
            'Binary-mask postprocessing cannot recover missing instance proposals']}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'selected_method':chosen,'depth_minus_spatial_APsmall':delta,
                      'variants':{k:{t:v[t] for t in ['APsmall_gain','AP_drop','eligible']} for k,v in audits.items()}}))


if __name__ == '__main__':
    main()
