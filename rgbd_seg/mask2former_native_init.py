"""Reverse the official Hugging Face Mask2Former Swin conversion, without I/O.

Mapping follows transformers v4.37.2's official checkpoint converter. Q/K/V
are concatenated in the original order; every source tensor must be accounted
for. This converts COCO initialization, never a trained DA-Fusion checkpoint.
"""
from __future__ import annotations
from collections import OrderedDict
import torch


def hf_to_native(state, config):
    output, used, mapping = {}, set(), {}

    def take(name):
        if name in used:
            raise ValueError('Source tensor used more than once: ' + name)
        used.add(name)
        return state[name]

    def copy(destination, source):
        if destination in output:
            raise ValueError('Duplicate native tensor: ' + destination)
        output[destination] = take(source)
        mapping[destination] = [source]

    def pair(destination, source):
        for suffix in ('weight', 'bias'):
            copy(destination + '.' + suffix, source + '.' + suffix)

    def qkv(destination, sources):
        for suffix in ('weight', 'bias'):
            names = [prefix + '.' + suffix for prefix in sources]
            # Swin uses qkv.weight; torch MultiheadAttention uses in_proj_weight.
            name = destination + ('.' if destination.endswith('qkv') else '_') + suffix
            output[name] = torch.cat([take(key) for key in names], dim=0)
            mapping[name] = names

    encoder = 'model.pixel_level_module.encoder'
    pair('backbone.patch_embed.proj', encoder + '.embeddings.patch_embeddings.projection')
    pair('backbone.patch_embed.norm', encoder + '.embeddings.norm')
    depths = config['backbone_config']['depths']
    for stage, depth in enumerate(depths):
        for block in range(depth):
            native = f'backbone.layers.{stage}.blocks.{block}'
            hf = f'{encoder}.encoder.layers.{stage}.blocks.{block}'
            for a, b in [('norm1', 'layernorm_before'), ('norm2', 'layernorm_after'),
                         ('attn.proj', 'attention.output.dense'),
                         ('mlp.fc1', 'intermediate.dense'), ('mlp.fc2', 'output.dense')]:
                pair(native + '.' + a, hf + '.' + b)
            for suffix in ('relative_position_bias_table', 'relative_position_index'):
                copy(native + '.attn.' + suffix, hf + '.attention.self.' + suffix)
            qkv(native + '.attn.qkv', [hf + '.attention.self.' + kind for kind in ('query', 'key', 'value')])
        if stage < len(depths) - 1:
            source = f'{encoder}.encoder.layers.{stage}.downsample'
            destination = f'backbone.layers.{stage}.downsample'
            copy(destination + '.reduction.weight', source + '.reduction.weight')
            pair(destination + '.norm', source + '.norm')
        pair(f'backbone.norm{stage}', f'{encoder}.hidden_states_norms.stage{stage+1}')

    pixel = 'model.pixel_level_module.decoder'
    native = 'sem_seg_head.pixel_decoder'
    for layer in ('adapter_1', 'layer_1'):
        copy(f'{native}.{layer}.weight', f'{pixel}.{layer}.0.weight')
        pair(f'{native}.{layer}.norm', f'{pixel}.{layer}.1')
    for level in range(3):
        for part in range(2):
            pair(f'{native}.input_proj.{level}.{part}', f'{pixel}.input_projections.{level}.{part}')
    copy(native + '.transformer.level_embed', pixel + '.level_embed')
    pair(native + '.mask_features', pixel + '.mask_projection')
    for layer in range(config['encoder_layers']):
        src = f'{pixel}.encoder.layers.{layer}'
        dst = f'{native}.transformer.encoder.layers.{layer}'
        for a, b in [('linear1', 'fc1'), ('linear2', 'fc2'),
                     ('norm1', 'self_attn_layer_norm'), ('norm2', 'final_layer_norm')]:
            pair(dst + '.' + a, src + '.' + b)
        for projection in ('attention_weights', 'output_proj', 'sampling_offsets', 'value_proj'):
            pair(dst + '.self_attn.' + projection, src + '.self_attn.' + projection)

    predictor = 'sem_seg_head.predictor'
    transformer = 'model.transformer_module'
    decoder = transformer + '.decoder'
    for layer in range(config['decoder_layers'] - 1):
        src = f'{decoder}.layers.{layer}'
        self_attn = f'{predictor}.transformer_self_attention_layers.{layer}'
        cross_attn = f'{predictor}.transformer_cross_attention_layers.{layer}'
        ffn = f'{predictor}.transformer_ffn_layers.{layer}'
        pair(self_attn + '.self_attn.out_proj', src + '.self_attn.out_proj')
        pair(self_attn + '.norm', src + '.self_attn_layer_norm')
        qkv(self_attn + '.self_attn.in_proj', [src + '.self_attn.' + p for p in ('q_proj', 'k_proj', 'v_proj')])
        for suffix in ('weight', 'bias'):
            copy(cross_attn + '.multihead_attn.in_proj_' + suffix, src + '.cross_attn.in_proj_' + suffix)
        pair(cross_attn + '.multihead_attn.out_proj', src + '.cross_attn.out_proj')
        pair(cross_attn + '.norm', src + '.cross_attn_layer_norm')
        for a, b in [('linear1', 'fc1'), ('linear2', 'fc2'), ('norm', 'final_layer_norm')]:
            pair(ffn + '.' + a, src + '.' + b)
    pair(predictor + '.decoder_norm', decoder + '.layernorm')
    for layer in range(3):
        pair(f'{predictor}.mask_embed.layers.{layer}', f'{decoder}.mask_predictor.mask_embedder.{layer}.0')
    for a, b in [('query_embed', 'queries_embedder'), ('query_feat', 'queries_features'), ('level_embed', 'level_embed')]:
        copy(predictor + '.' + a + '.weight', transformer + '.' + b + '.weight')
    pair(predictor + '.class_embed', 'class_predictor')
    ignored = set(state) - used
    if ignored != {'criterion.empty_weight'}:
        raise ValueError('Unaccounted or missing tensors: ' + repr(sorted(ignored)))
    return output, {'source_tensors': len(state), 'converted_tensors': len(output),
                    'ignored_nonparameter_buffers': sorted(ignored), 'mapping': mapping}


def duplicate_native_backbones(state, *, drop_queries=True):
    """Apply the public DA-Fusion remap policy; fusion and adapter stay fresh.

Class logits always reset for one foreground class. The public script also
drops query_feat/query_embed; make that choice explicit in the receipt.
    """
    output, dropped = OrderedDict(), []
    for name, value in state.items():
        if 'class_embed' in name or (drop_queries and any(k in name for k in ('query_feat', 'query_embed', 'static_query'))):
            dropped.append(name)
        elif name.startswith('backbone.'):
            suffix = name.removeprefix('backbone.')
            output['backbone.swin_rgb.' + suffix] = value
            output['backbone.swin_d.' + suffix] = value
        else:
            output[name] = value
    # MaskFormerHead._load_from_state_dict treats absent metadata as version 1
    # and would insert a SECOND pixel_decoder prefix into already current keys.
    # Preserve standard PyTorch state-dict metadata through torch.save/load.
    output._metadata = {'sem_seg_head': {'version': 2},
                        'sem_seg_head.predictor': {'version': 2}}
    return output, {'drop_queries': drop_queries, 'dropped': dropped,
                    'state_dict_metadata': output._metadata,
                    'depth_stem': 'copy RGB three-channel stem; depth adapter handles XYZ+validity',
                    'initialized_tensors': len(output)}
