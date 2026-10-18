#!/usr/bin/env python3
"""Audit continued original point heads, including their restored starting weights."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from rgbd_seg.course_archive import CourseArchive
from rgbd_seg.course_distributed_eval import write_json
from rgbd_seg.course_optimized import file_sha, tensor_digest
from tools.audit_course_point_refinement import inspect_rows
from tools.run_course_point_continuation import select_validation, selection_history, parent_inputs
from tools.run_course_point_refinement import VARIANTS, checked, now, read, require


def audit(protocol, supervisor_path):
    supervisor = read(supervisor_path)
    root = supervisor_path.parent / 'experiment'
    require(supervisor['status'] == 'completed' and supervisor['exit_code'] == 0,
            'Training supervisor did not complete')
    require(read(supervisor_path.parent / 'exit.json') == {'status': 'completed', 'exit_code': 0},
            'Exit receipt differs')
    require(supervisor['protocol_sha256'] == protocol['_sha256'], 'Supervisor protocol differs')
    require(protocol['evaluation_policy'] == 'validation_only', 'Expected validation-only run')
    require(not (root / 'test').exists(), 'Unexpected test outputs')
    require(read(root / 'run.json')['status'] == 'completed', 'Trainer is incomplete')
    final = read(root / 'comparison.json')
    require(final['status'] == 'completed' and final['scope'] == protocol['scope']
            and final['protocol_sha256'] == protocol['_sha256'] and not final['test_used'],
            'Final report identity or scope differs')
    artifacts = {protocol['_path']: protocol['_sha256']}
    for name, digest in supervisor['artifact_sha256'].items():
        require(file_sha(name) == digest, f'Changed artifact: {name}')
        artifacts[name] = digest
    world, epochs = len(protocol['gpus']), protocol['epochs']
    parent_epoch=protocol['initial_head']['checkpoint']['epoch']
    _,parent_validation=parent_inputs(protocol)
    initial=read(root/'initial_head_reload.json')
    require(initial['epoch']==parent_epoch and not initial['original_optimizer_state_available']
            and initial['checkpoint_sha256']==protocol['initial_head']['checkpoint']['sha256']
            and initial['head_tensor_sha256']==parent_validation['checkpoint']['head_tensor_sha256'],
            'Initial heads were not restored from the validated parent')
    require(protocol['global_batch_size'] == world * protocol['batch_size'], 'Batch layout differs')
    steps_per_epoch = protocol['training_images'] // protocol['global_batch_size']
    expected_steps = epochs * steps_per_epoch
    with CourseArchive(protocol['archive'], 'train') as data:
        require(len(data) == 14432, 'Official training split differs')
        indices = protocol.get('training_indices', list(range(len(data))))
        train_ids = [data.image_ids[index] for index in indices]
    require(len(train_ids) == protocol['training_images'], 'Protocol training size differs')
    with CourseArchive(protocol['archive'], 'val') as data:
        require(len(data) == 2584, 'Official validation split differs')
        val_ids = data.image_ids if protocol['scope'] == 'full_experiment' else data.image_ids[:2 * world]
    history = read(root / 'validation_history.json')
    require(len(history) == epochs and history == final['validation_history'], 'History is incomplete')
    base = read(root / 'base_reload.json')
    require(base['checkpoint_sha256'] == protocol['weights']['sha256'], 'Base checkpoint differs')
    rank_receipts = [read(root / f'completed_rank{rank}.json') for rank in range(world)]
    for rank, receipt in enumerate(rank_receipts):
        require(receipt['status'] == 'completed' and receipt['rank'] == rank
                and receipt['global_steps'] == expected_steps and receipt['base_unchanged']
                and receipt['base_gradients_absent']
                and receipt['base_tensor_sha256_before_and_after'] == base['model_sha256'],
                'Rank completion or frozen base verification differs')
    require(len({r['head_tensor_sha256'] for r in rank_receipts}) == 1, 'Rank head weights differ')
    require(final['rank_receipts'] == rank_receipts, 'Final rank evidence differs')
    checkpoint_digests = []
    for epoch in range(parent_epoch + 1, parent_epoch + epochs + 1):
        ids = [item for rank in range(world)
               for item in read(root / f'coverage_epoch{epoch:02d}_rank{rank}.json')]
        require(len(ids) == len(set(ids)) == len(train_ids) and set(ids) == set(train_ids),
                'Training examples missing or duplicated')
        for rank in range(world):
            path = root / f'train_epoch{epoch:02d}_rank{rank}.jsonl'
            import json
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            require([row['step'] for row in rows] == list(range(1, steps_per_epoch + 1)),
                    'Training update coverage differs')
            for step, row in enumerate(rows, 1):
                require(row['epoch'] == epoch and row['global_step'] == (epoch - parent_epoch - 1) * steps_per_epoch + step,
                        'Training update identity differs')
                require(row['global_instances'] > 0 and all(math.isfinite(row[key])
                        for key in ('point_loss', 'depth_point_loss', 'lr', 'seconds')), 'Non-finite training log')
            artifacts[str(path)] = file_sha(path)
        checkpoint = read(root / f'checkpoint_epoch_{epoch:02d}.json')
        require(checkpoint['epoch'] == epoch and file_sha(checkpoint['path']) == checkpoint['sha256'],
                'Epoch checkpoint identity differs')
        values = torch.load(checkpoint['path'], map_location='cpu', weights_only=True)
        require(values['protocol_sha256'] == protocol['_sha256']
                and values['source_sha256'] == protocol['source_sha256']
                and values['point_policy'] == protocol['point_policy']
                and values['base_checkpoint_sha256'] == protocol['weights']['sha256'],
                'Checkpoint provenance differs')
        digest = tensor_digest(values['heads'])
        require(digest == checkpoint['head_tensor_sha256'], 'Head tensor digest differs')
        require(all(torch.isfinite(value).all() for value in values['heads'].values()), 'Non-finite head tensor')
        checkpoint_digests.append(digest)
        folder = root / f'val_epoch_{epoch:02d}'
        require(read(folder / 'image_ids.json') == val_ids, 'Validation IDs differ')
        inspect_rows(folder, val_ids, artifacts)
        require(history[epoch - parent_epoch - 1] == read(folder / 'comparison.json'), 'Validation history binding differs')
        for variant in VARIANTS:
            require(history[epoch - parent_epoch - 1]['metrics'][variant] == read(folder / variant / 'evaluation.json')['metrics'],
                    'Reported metrics differ from evaluator outputs')
    if epochs > 1:
        require(len(set(checkpoint_digests)) > 1, 'Heads did not change between training epochs')
    selection = select_validation(selection_history(protocol, history))
    require(selection == read(root / 'selection.json') == final['selection'], 'APs selection policy differs')
    require(rank_receipts[0]['head_tensor_sha256'] == selection['selected']['checkpoint']['head_tensor_sha256'],
            'Selected checkpoint was not reloaded')
    # This trusted local checkpoint contains Python/NumPy RNG states as well as tensors.
    training = torch.load(root / 'last_training.pt', map_location='cpu', weights_only=False)
    require(training['epoch'] == parent_epoch + epochs and training['global_step'] == expected_steps
            and training['protocol_sha256'] == protocol['_sha256']
            and training['source_sha256'] == protocol['source_sha256']
            and training['base_checkpoint_sha256'] == protocol['weights']['sha256']
            and training['validation_history'] == history, 'Saved training state provenance differs')
    require(tensor_digest(training['heads']) == checkpoint_digests[-1], 'Last training state is not the last epoch')
    require(tensor_digest(training['heads']) != initial['head_tensor_sha256'], 'Training did not update the initial heads')
    states = training['optimizer']['state']
    require(states and all(int(value['step']) == expected_steps for value in states.values()),
            'Optimizer state is incomplete')
    require(all(torch.isfinite(value[key]).all() for value in states.values()
                for key in ('exp_avg', 'exp_avg_sq')), 'Non-finite optimizer moments')
    require(len(training['rank_rng']) == world and all(training['rank_rng']), 'Missing rank RNG state')
    artifacts[str(root / 'last_training.pt')] = file_sha(root / 'last_training.pt')
    return {'status': 'passed', 'at_utc': now(), 'scope': protocol['scope'],
            'protocol_sha256':protocol['_sha256'], 'parent_head_epoch':parent_epoch,
            'initial_head_restored':True,'optimizer_initialization':'fresh AdamW',
            'material_passport': {'origin_skill': 'academic-research-suite/experiment-agent', 'mode': 'run',
                                 'verification_status': 'VALIDATION_ARTIFACTS_AUDITED'},
            'world_size': world, 'epochs': epochs, 'training_images_per_epoch': len(train_ids),
            'validation_images_per_epoch': len(val_ids), 'updates': expected_steps,
            'base_frozen': True, 'selected_head_reloaded': True, 'optimizer_state_saved': True,
            'rng_states_saved': world, 'test_used': False, 'artifact_sha256': artifacts,
            'limitations': ['Engineering subsets do not establish model performance',
                            'This audit does not claim exact training resumption has been exercised',
                            'Single-GPU continuation preserves global batch eight but changes rank layout and sampling RNG']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--supervisor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(checked(args.protocol), args.supervisor)
    write_json(args.output, result)
    print({key: result[key] for key in ('status', 'scope', 'world_size', 'epochs', 'updates')})


if __name__ == '__main__':
    main()
