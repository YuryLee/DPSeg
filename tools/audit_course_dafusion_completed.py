#!/usr/bin/env python3
"""Audit saved completed DA-Fusion artifacts without model execution or rescoring."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '4'

from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from zipfile import ZipFile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rgbd_seg.course_archive import CourseArchive

FORMAL = ROOT/'reports/course_dafusion/formal_v1'
OUTPUT = ROOT/'reports/course_dafusion/final_artifact_audit_core.json'
EVIDENCE = {}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path, label):
    path = Path(path)
    value = path.read_bytes()
    EVIDENCE[label] = {'path': str(path.resolve()), 'sha256': hashlib.sha256(value).hexdigest(), 'bytes': len(value)}
    return json.loads(value)


def canon(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def source_check(values, short_names=False):
    for name, expected in values.items():
        path = ROOT/('rgbd_seg' if short_names and '/' not in name else '')/name
        assert digest(path) == expected, name


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    protocol = read_json(ROOT/'reports/course_dafusion/protocol.json', 'protocol')
    run = read_json(FORMAL/'run.json', 'run')
    initialization = read_json(FORMAL/'initialization.json', 'initialization')
    selection = read_json(FORMAL/'selection.json', 'selection')
    protocol_sha = EVIDENCE['protocol']['sha256']
    assert protocol_sha == '7901485e6e50d125bfd7bcc1c01e3c50876782d5013869248771ed94868915e9'
    assert protocol['status'] == 'frozen' and run['status'] == 'completed'
    assert run['protocol_sha256'] == protocol_sha
    assert run['source_sha256'] == protocol['source_sha256'] and len(run['source_sha256']) == 28
    source_check(protocol['source_sha256'])
    assert digest(protocol['execution_driver']['path']) == protocol['execution_driver']['sha256']
    assert canon(initialization) == canon(run['model_metadata'])
    for key in ('epochs','batch_size','accumulation','seed','lr','precision','workers','threads'):
        assert run['arguments'][key] == protocol['training'][key], key
    assert run['arguments']['batch_size'] == 4 and run['arguments']['accumulation'] == 2
    assert run['arguments']['smoke_steps'] == 0 and run['arguments']['resume'] is False
    assert run['optimizer_steps'] == run['expected_optimizer_steps'] == 9020
    assert run['training_images_seen'] == run['expected_training_images_seen'] == 72160
    assert run['effective_batch_size'] == 8 and run['depth_used'] is True
    assert run['test_used_for_checkpoint_selection'] is False
    assert run['elapsed_seconds'] < protocol['limits']['per_process_timeout_seconds']
    audit = read_json(protocol['data_audit']['path'], 'data_audit')
    assert EVIDENCE['data_audit']['sha256'] == protocol['data_audit']['sha256']
    pretrained = Path(run['arguments']['checkpoint'])/'model.safetensors'
    assert digest(pretrained) == protocol['models']['dafusion']['initial_sha256'] == run['verified_input_sha256']['pretrained_sha256']
    assert run['verified_input_sha256']['archive_sha256'] == protocol['archive_sha256_from_completed_download_receipt']
    with CourseArchive(protocol['archive'], split='train') as data:
        official_train_ids = data.image_ids
        assert len(official_train_ids) == 14432
        h = hashlib.sha256()
        with ZipFile(data.zip_path) as archive:
            for name in data.annotation_members:
                h.update(name.encode()+b'\0'+archive.read(name))
        assert h.hexdigest() == audit['splits']['train']['annotation_member_name_and_bytes_sha256']
        train_annotation_sha = h.hexdigest()
    assert run['training_data']['annotation_member_name_and_bytes_sha256'] == train_annotation_sha
    raw = (FORMAL/'training_log.jsonl').read_bytes()
    lines = raw.splitlines(keepends=True)
    assert len(lines) == 9020 and all(line.endswith(b'\n') for line in lines)
    rows = [json.loads(line) for line in lines]
    assert [row['global_step'] for row in rows] == list(range(1, 9021))
    assert all(row['total_steps'] == 9020 and row['expected_train_images'] == 14432 for row in rows)
    for field in ('loss','grad_norm_before_clip','head_lr','step_seconds_with_data'):
        assert all(math.isfinite(row[field]) for row in rows), field
    assert all(row['grad_norm_before_clip'] >= 0 for row in rows)
    for index, row in enumerate(rows):
        factor = .1+.9*(index+1)/250 if index < 250 else .1+.9*.5*(1+math.cos(math.pi*(index-250)/8769))
        assert abs(row['head_lr'] - .0002*factor) < 1e-15
    EVIDENCE['training_log'] = {'path':str(FORMAL/'training_log.jsonl'),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    assert [e['epoch'] for e in selection['epochs']] == [1,2,3,4,5]
    assert selection['test_used_for_selection'] is False and selection['validation_scope'] == 'full_2584_images'
    epochs = []
    val_reference = None
    for epoch in range(1,6):
        records = rows[(epoch-1)*1804:epoch*1804]
        assert all(row['epoch'] == epoch for row in records)
        assert [row['images_in_epoch'] for row in records] == list(range(8,14433,8))
        mean_loss = sum(row['loss']*8 for row in records)/14432
        entry = selection['epochs'][epoch-1]
        assert entry['global_step'] == epoch*1804 and entry['training_images'] == 14432
        assert abs(mean_loss-entry['training_loss_mean']) < 1e-10
        order = read_json(FORMAL/f'train_order_epoch_{epoch:02d}.json', f'train_order_epoch{epoch}')
        expected = [official_train_ids[i] for i in torch.randperm(14432, generator=torch.Generator().manual_seed(17+1009*(epoch-1))).tolist()]
        assert order['epoch'] == epoch and order['count'] == 14432
        assert order['image_ids'] == expected and len(set(expected)) == 14432
        assert order['ordered_image_ids_sha256'] == hashlib.sha256(json.dumps(expected).encode()).hexdigest()
        for baseline in ('maskrcnn','mask2former'):
            baseline_order = read_json(ROOT/f'reports/course_finetune/{baseline}/train_order_epoch_{epoch:02d}.json', f'{baseline}_order_epoch{epoch}')
            assert baseline_order == order
        folder = FORMAL/f'val_epoch_{epoch:02d}'
        evaluation = check_evaluation(folder, 'val', 2584, f'val_epoch{epoch}')
        assert evaluation['metrics'] == entry['metrics'] and Path(entry['output']).resolve() == folder
        settings = metric_settings(evaluation)
        if val_reference is None:
            val_reference = settings
        else:
            assert settings == val_reference
        epochs.append({'epoch':epoch,'updates':1804,'images':14432,'all_ids_once':True,'same_order_as_both_rgb_baselines':True,
                       'loss_mean':mean_loss,'loss_min':min(r['loss'] for r in records),'loss_max':max(r['loss'] for r in records),
                       'preclip_gradient_max':max(r['grad_norm_before_clip'] for r in records),'metrics':evaluation['metrics']})
    assert val_reference['gt_instances'] == audit['splits']['val']['annotations']
    assert val_reference['gt_small_including_crowd'] == audit['splits']['val']['small_gt_area_le_1024']
    chosen = max(selection['epochs'], key=lambda e:(e['metrics']['mask_ap_small'],e['metrics']['mask_ap'],-e['epoch']))
    assert chosen['epoch'] == selection['best_epoch'] == run['best_epoch'] == 5
    assert selection['best_key'] == [chosen['metrics']['mask_ap_small'],chosen['metrics']['mask_ap']]
    test = check_evaluation(FORMAL/'test', 'test', 2112, 'test')
    test_receipt = json.loads((FORMAL/'test/evaluation_run.json').read_text())
    assert run['test_metrics'] == test['metrics']
    assert EVIDENCE['test_run']['sha256'] == run['test_evaluation_run_sha256']
    assert Path(run['test_output']).resolve() == FORMAL/'test'
    best_sha, last_sha = digest(FORMAL/'best.pt'), digest(FORMAL/'last.pt')
    assert best_sha == run['best_checkpoint_sha256'] == test_receipt['evaluated_checkpoint_sha256']
    assert test_receipt['selected_epoch'] == 5
    print(json.dumps({'status':'loading_saved_checkpoint_metadata_on_cpu'}), flush=True)
    best = torch.load(FORMAL/'best.pt', map_location='cpu', weights_only=True, mmap=True)
    last = torch.load(FORMAL/'last.pt', map_location='cpu', weights_only=False, mmap=True)
    for checkpoint in (best,last):
        assert checkpoint['source_sha256'] == protocol['source_sha256'] and checkpoint['protocol_sha256'] == protocol_sha
        assert canon(checkpoint['model_metadata']) == canon(initialization)
    assert best['model_name'] == 'dafusion' and best['epoch'] == 5 and best['selection'] == chosen
    state = last['state']
    assert (state['epoch'],state['images_in_epoch'],state['global_step'],state['stage']) == (5,0,9020,'training')
    assert state['val_results'] == selection['epochs'] and state['best_key'] == selection['best_key'] and state['best_epoch'] == 5
    assert state['epoch_loss_sum'] == 0 and canon(last['training_data']) == canon(run['training_data'])
    assert best['model'].keys() == last['model'].keys()
    for name, value in best['model'].items():
        assert torch.isfinite(value).all() and torch.equal(value, last['model'][name]), name
    optimizer_values = 0
    for values in last['optimizer']['state'].values():
        for value in values.values():
            if torch.is_tensor(value):
                assert torch.isfinite(value).all()
                optimizer_values += value.numel()
        assert 0 < float(values['step']) <= 9020
    for group in last['optimizer']['param_groups']:
        assert group['base_lr'] in (2e-5,2e-4) and group['weight_decay'] in (0.,.05)
        assert abs(group['lr'] - .1*group['base_lr']) < 1e-16
    checkpoint_result = {'best_epoch':5,'best_sha256':best_sha,'last_sha256':last_sha,'model_state_tensors':len(best['model']),
        'best_and_last_tensors_exactly_equal':True,'all_model_and_optimizer_tensors_finite':True,
        'optimizer_numeric_elements_checked':optimizer_values,'test_receipt_bound_to_actual_best_file':True,
        'last_resume_state':{k:v for k,v in state.items() if k!='val_results'}}
    del best,last
    gc.collect()
    EVIDENCE['best_checkpoint'] = {'path':str(FORMAL/'best.pt'),'sha256':best_sha}
    EVIDENCE['last_checkpoint'] = {'path':str(FORMAL/'last.pt'),'sha256':last_sha}
    assert sorted(p.name for p in FORMAL.glob('test*')) == ['test']
    assert not list(FORMAL.glob('*attempt*')) and not list(FORMAL.glob('resume_*'))
    assert sorted(p.name for p in FORMAL.glob('val_epoch_*')) == [f'val_epoch_{e:02d}' for e in range(1,6)]
    console_path = ROOT/'reports/course_dafusion/launch_v1/console.log'
    console = console_path.read_text()
    events=[]
    for line in console.splitlines():
        if line.startswith('{'):
            try:events.append(json.loads(line))
            except json.JSONDecodeError:pass
    assert 'Traceback (most recent call last)' not in console
    stages = Counter(e.get('status') for e in events if e.get('phase') == 'testing')
    for key in ('warming_up','validating_submission','evaluating','completed'):
        assert stages[key] == 1, (key,stages)
    final_events = [e for e in events if e.get('model') == 'dafusion' and e.get('status') == 'completed']
    assert len(final_events) == 1 and final_events[0]['test_metrics'] == test['metrics']
    EVIDENCE['console'] = {'path':str(console_path),'sha256':digest(console_path)}
    observation = read_json(ROOT/'reports/course_dafusion/process_completion_observation.json', 'process_observation')
    assert observation['process_exit_code'] is None
    assert not (ROOT/'reports/course_dafusion/launch_v1/exit.json').exists()
    baseline_comparison={}
    for name in ('maskrcnn','mask2former'):
        item = read_json(ROOT/f'reports/course_finetune/{name}/test/evaluation.json', f'{name}_test_evaluation')
        assert metric_settings(item) == metric_settings(test)
        baseline_comparison[name]={'mask_ap_small':item['metrics']['mask_ap_small'],'mask_ap':item['metrics']['mask_ap']}
    source_check(protocol['source_sha256'])
    assert not torch.cuda.is_initialized()
    report={'status':'pass','scope':'completed_training_validation_test_artifact_core_audit; CSV native audit is separate',
        'audited_at_utc':datetime.now(timezone.utc).isoformat(),'auditor_sha256':digest(__file__),
        'frozen_sources_unchanged':28,'protocol_sha256':protocol_sha,'epochs':epochs,'updates':9020,'training_image_exposures':72160,
        'all_training_losses_and_gradients_finite':True,'lr_schedule_verified_for_all_9020_updates':True,
        'train_annotation_member_digest':train_annotation_sha,'archive_full_sha256_recomputed_here':False,
        'original_training_archive_sha256':run['verified_input_sha256']['archive_sha256'],
        'checkpoint':checkpoint_result,'test_metrics':test['metrics'],'baseline_test_metrics':baseline_comparison,
        'recorded_test_evaluations':1,'test_phase_counts':dict(stages),'resumed_runs_or_attempt_directories':False,
        'gpu_used_by_audit':False,'model_forward_or_scoring_run':False,
        'provenance_limitations':[
            'Supervisor exit.json is absent and the operating-system exit code was not captured; do not claim exit code 0. Root process observation is linked separately; file completion status is not process liveness evidence.',
            'Frozen protocol prose training.gradient_accumulation says batch 2 x 4. Numeric protocol fields, launch command, run arguments and optimizer budget consistently show batch 4 x accumulation 2 = 8. Frozen protocol was not edited.',
            'This checks existing artifacts and saved tensors; it is not an independent training or COCO score reproduction.'],
        'evidence':EVIDENCE,'elapsed_seconds':time.monotonic()-started}
    OUTPUT.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'pass','output':str(OUTPUT),'sha256':digest(OUTPUT),'elapsed_seconds':report['elapsed_seconds']}),flush=True)


def metric_settings(value):
    return {key:value[key] for key in ('iou_type','max_detections','iou_thresholds','area_ranges','gt_instances',
        'gt_small_including_crowd','gt_area_groups_including_crowd','ground_truth_area','prediction_area')}


def check_evaluation(folder, split, expected_images, label):
    evaluation = read_json(folder/'evaluation.json',label+'_evaluation')
    receipt = read_json(folder/'evaluation_run.json',label+'_run')
    assert evaluation['status'] == receipt['status'] == 'completed'
    assert evaluation['scope'] == receipt['scope'] == 'full_split' and evaluation['split'] == receipt['split'] == split
    assert evaluation['images'] == receipt['images'] == evaluation['full_split_images'] == receipt['full_split_images'] == expected_images
    assert evaluation['depth_used'] is True and receipt['depth_used'] is True
    assert len(evaluation['metrics']) == 12 and all(math.isfinite(v) and 0 <= v <= 1 for v in evaluation['metrics'].values())
    assert evaluation['iou_type'] == 'segm' and evaluation['max_detections'] == [1,10,100]
    assert all(abs(a-(.5+.05*i))<1e-15 for i,a in enumerate(evaluation['iou_thresholds'])) and len(evaluation['iou_thresholds'])==10
    assert evaluation['area_ranges'] == [[0,1e10],[0,1024],[1024,9216],[9216,1e10]]
    assert receipt['evaluation_sha256'] == EVIDENCE[label+'_evaluation']['sha256']
    prediction_path=folder/'predictions.jsonl';prediction_sha=digest(prediction_path)
    assert prediction_sha == evaluation['predictions_sha256'] == receipt['predictions_sha256']
    with CourseArchive(receipt['archive'],split=split) as data:
        expected_ids=data.image_ids
    ids=[];count=0
    with prediction_path.open() as stream:
        for line in stream:
            item=json.loads(line);ids.append(item['image_id']);count+=len(item['predictions'])
            assert len(item['predictions'])<=100
    assert ids == expected_ids and len(ids)==len(set(ids))==expected_images
    assert count == evaluation['predicted_instances']
    source_check(evaluation['source_sha256'],short_names=True)
    assert receipt['source_sha256'] == evaluation['source_sha256']
    EVIDENCE[label+'_predictions']={'path':str(prediction_path),'sha256':prediction_sha,'rows':len(ids),'instances':count}
    return evaluation


if __name__ == '__main__':
    main()
