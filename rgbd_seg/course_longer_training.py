"""Continuation schedules and validation of the completed course checkpoint."""
from __future__ import annotations

import math


def continuation_lr(step, total_steps, *, peak_lr, start_lr, warmup_steps, minimum_ratio):
    if not (0 <= step < total_steps and 0 < start_lr <= peak_lr
            and 0 < minimum_ratio <= 1 and 0 <= warmup_steps < total_steps):
        raise ValueError('Invalid continuation learning-rate schedule')
    if warmup_steps and step < warmup_steps:
        return start_lr + (peak_lr-start_lr)*step/max(1, warmup_steps-1)
    progress = (step-warmup_steps)/max(1, total_steps-warmup_steps-1)
    return peak_lr*(minimum_ratio+(1-minimum_ratio)*.5*(1+math.cos(math.pi*progress)))


def check_parent_checkpoint(checkpoint, parent_protocol, parent_sha, metadata, expected_epoch):
    if (checkpoint['protocol_sha256'] != parent_sha
            or checkpoint['source_sha256'] != parent_protocol['source_sha256']
            or checkpoint['model_metadata'] != metadata):
        raise ValueError('Parent checkpoint construction or provenance differs')
    state = checkpoint['state']
    expected_steps = math.ceil(14432/parent_protocol['training']['effective_batch_size'])*expected_epoch
    if (state['epoch'] != expected_epoch or state['images_in_epoch'] != 0
            or state['stage'] != 'training' or state['global_step'] != expected_steps
            or len(state['val_results']) != expected_epoch):
        raise ValueError('Parent must be a fully completed, validated epoch')
    if state['best_epoch'] != expected_epoch:
        raise ValueError('This experiment starts from the validated best final epoch')
    return state['val_results'][-1]
