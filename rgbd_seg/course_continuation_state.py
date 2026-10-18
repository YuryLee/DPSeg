"""Validate absolute counters before continuing a completed course run again.

This module is separate from the frozen v1 runner. A checkpoint check alone
does not establish successful process exit or replace the completed-run audit.
"""
from __future__ import annotations

import math

import torch


def validated_parent_state(checkpoint, protocol, protocol_sha, metadata, expected_epoch):
    """Return a validated final-epoch candidate and cumulative training offsets.

Resolution runs use absolute local counters. Continuation runs reset their local
epoch/update counters and store the previous offsets in their protocol. Only a
validated best final epoch is accepted for the planned convergence extension.
"""
    if (checkpoint['protocol_sha256'] != protocol_sha
            or checkpoint['source_sha256'] != protocol['source_sha256']
            or checkpoint['model_metadata'] != metadata):
        raise ValueError('Parent checkpoint provenance or model construction differs')
    schema = protocol['schema']
    if schema == 'course_resolution_ddp_v1':
        epoch_offset, update_offset = 0, 0
    elif schema in ('course_longer_training_v1', 'course_longer_extension_v1'):
        epoch_offset = protocol['continuation']['parent_epoch']
        update_offset = protocol['continuation']['parent_optimizer_steps']
        if not (isinstance(epoch_offset, int) and isinstance(update_offset, int)
                and epoch_offset > 0 and update_offset > 0):
            raise ValueError('Invalid parent continuation offsets')
    else:
        raise ValueError(f'Unsupported parent training schema: {schema}')
    train, state = protocol['training'], checkpoint['state']
    epochs = train['epochs']
    updates = math.ceil(14432 / train['effective_batch_size'])
    local_updates = epochs * updates
    total_epoch, total_updates = epoch_offset + epochs, update_offset + local_updates
    if (train['images_per_epoch'] != 14432 or train['updates_per_epoch'] != updates
            or train['total_updates'] != local_updates or total_epoch != expected_epoch):
        raise ValueError('Parent protocol training counts differ')
    if (state['epoch'] != epochs or state['global_step'] != local_updates
            or state['images_in_epoch'] != 0 or state['stage'] != 'training'
            or len(state['val_results']) != epochs):
        raise ValueError('Parent checkpoint is not a completed final epoch')
    for index, entry in enumerate(state['val_results'], 1):
        if (entry['epoch'] != epoch_offset + index or entry['global_step'] != index * updates
                or entry['training_images'] != 14432):
            raise ValueError('Parent validation history counters differ')
        if not all(isinstance(entry['metrics'][key], (int, float))
                   and math.isfinite(entry['metrics'][key]) and 0 <= entry['metrics'][key] <= 1
                   for key in ('mask_ap', 'mask_ap_small')):
            raise ValueError('Parent validation has an invalid AP metric')
    selection = state['val_results'][-1]
    key = [selection['metrics']['mask_ap_small'], selection['metrics']['mask_ap']]
    if (state['best_epoch'] != total_epoch or state['best_key'] != key
            or any((v['metrics']['mask_ap_small'], v['metrics']['mask_ap'], -v['epoch'])
                   > (key[0], key[1], -total_epoch) for v in state['val_results'])):
        raise ValueError('Parent final epoch is not the validation-selected best')
    optimizer = checkpoint['optimizer']
    states = optimizer['state']
    parameter_ids = [p for group in optimizer['param_groups'] for p in group['params']]
    if (not states or len(parameter_ids) != len(set(parameter_ids))
            or not set(states).issubset(parameter_ids)):
        raise ValueError('Parent optimizer parameter coverage differs')
    for value in states.values():
        step = value['step']
        if not torch.isfinite(torch.as_tensor(step)).all() or float(step) != total_updates:
            raise ValueError('Parent AdamW step is not the cumulative update count')
        if (value['exp_avg'].shape != value['exp_avg_sq'].shape
                or not torch.isfinite(value['exp_avg']).all()
                or not torch.isfinite(value['exp_avg_sq']).all()):
            raise ValueError('Parent AdamW moments are invalid')
    if len(checkpoint['rank_rng']) != len(protocol['hardware']['physical_gpus']):
        raise ValueError('Parent rank RNG state coverage differs')
    return {'parent_epoch': total_epoch, 'parent_optimizer_steps': total_updates,
            'local_epochs': epochs, 'local_optimizer_steps': local_updates,
            'inherited_epoch_offset': epoch_offset, 'inherited_update_offset': update_offset,
            'selection': selection, 'optimizer_parameters_with_state': len(states)}
