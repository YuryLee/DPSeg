import numpy as np
import unittest
import torch
from pycocotools import mask as mask_utils

from rgbd_seg.course_depth_data import depth_to_xyz
from rgbd_seg.course_depth_refinement import refine_masks
from rgbd_seg.course_finetune_mask2former import component_predictions
from rgbd_seg.course_tiled_probability import (
    Candidate, candidates_from_logits, crop_sample, encode_mask, merge_candidates,
    refine_probability_masks, view_bounds,
)


def test_crop_preserves_physical_camera_coordinates_and_alignment():
    h, w = 12, 15
    depth = torch.linspace(.3, .7, h*w).reshape(1, h, w)
    valid = torch.ones_like(depth, dtype=torch.bool)
    sample = {'image': torch.arange(3*h*w).reshape(3, h, w).float(),
              'depth_m': depth, 'depth_valid': valid,
              'camera_K': torch.tensor([[100., 0., 7.], [0., 110., 6.], [0., 0., 1.]])}
    cropped = crop_sample(sample, (4, 3, 11, 10))
    expected = depth_to_xyz(depth, valid, sample['camera_K'])[:, 3:10, 4:11]
    actual = depth_to_xyz(cropped['depth_m'], cropped['depth_valid'], cropped['camera_K'])
    torch.testing.assert_close(actual, expected)
    assert torch.equal(cropped['image'], sample['image'][:, 3:10, 4:11])
    coverage = np.zeros((1024, 1024), dtype=int)
    for left, top, right, bottom in view_bounds(1024, 1024)[1:]:
        coverage[top:bottom, left:right] += 1
    assert coverage.min() >= 1 and coverage.max() == 4


def test_raw_probability_extraction_matches_frozen_adapter():
    generator = torch.Generator().manual_seed(19)
    classes = torch.randn(17, 2, generator=generator)
    logits = torch.randn(17, 7, 7, generator=generator)
    sizes = dict(padded_size=(32, 32), image_size=(27, 29), original_size=(39, 41))
    expected = component_predictions(classes, logits, **sizes)
    actual = candidates_from_logits(classes, logits, **sizes, bounds=(0, 0, 41, 39),
                                    full_size=(39, 41), view_id=0)
    assert [c.public() for c in actual] == [{k: p[k] for k in ('category_id', 'score', 'segmentation')}
                                          for p in expected]


def test_soft_values_can_recover_low_confidence_boundary_pixel():
    soft = torch.full((1, 7, 7), .1)
    soft[:, :, 2:] = .9
    soft[0, 3, 2] = .49
    original = soft > .5
    depth = torch.ones(7, 7)
    valid = torch.ones(7, 7, dtype=torch.bool)
    refined, _ = refine_probability_masks(soft, original, depth, valid, radius=1)
    assert refined[0, 3, 2]
    # A physical depth discontinuity prevents averaging across this boundary.
    depth[:, :2] = 2.
    guarded, _ = refine_probability_masks(soft, original, depth, valid, radius=1)
    assert not guarded[0, 3, 1]


def test_binary_probabilities_reproduce_existing_depth_refiner():
    generator = torch.Generator().manual_seed(3)
    masks = torch.rand((5, 13, 15), generator=generator) > .6
    depth = .5+torch.rand((13, 15), generator=generator)*.004
    valid = torch.rand((13, 15), generator=generator) > .1
    depth[~valid] = float('nan')
    expected, stats = refine_masks(masks, depth, valid, mode='depth')
    actual, other = refine_probability_masks(masks.float(), masks, depth, valid)
    assert torch.equal(actual, expected) and other == stats


def test_invalid_depth_and_empty_result_preserve_original():
    original = torch.zeros(1, 7, 7, dtype=torch.bool)
    original[0, 3, 3] = True
    depth = torch.ones(7, 7)
    soft = torch.full((1, 7, 7), .01)
    actual, stats = refine_probability_masks(soft, original, depth, torch.ones(7, 7, dtype=torch.bool))
    assert torch.equal(actual, original) and stats['empty_reverted'] == 1
    valid = torch.zeros(7, 7, dtype=torch.bool)
    actual, _ = refine_probability_masks(soft, original, depth*float('nan'), valid)
    assert torch.equal(actual, original)


def test_mask_nms_removes_duplicate_view_but_retains_adjacent_object():
    a, b = np.zeros((16, 16), bool), np.zeros((16, 16), bool)
    a[2:9, 3:8] = True
    b[2:9, 8:12] = True
    def candidate(mask, score, view):
        return Candidate(score, 0, view, (0, 0, 16, 16), torch.from_numpy(mask).float(), encode_mask(mask))
    selected, stats = merge_candidates([candidate(a, .8, 0), candidate(a, .9, 1), candidate(b, .7, 1)])
    assert len(selected) == 2 and stats['duplicates_removed'] == 1
    assert selected[0].view_id == 1
    assert np.array_equal(mask_utils.decode(selected[1].segmentation), b)


def test_internal_crop_boundary_filter_preserves_outer_image_boundary():
    classes = torch.tensor([[8., -8.]])
    masks = torch.full((1, 8, 8), -8.)
    masks[:, :, :2] = 8.
    kwargs = dict(padded_size=(8, 8), image_size=(8, 8), original_size=(8, 8), full_size=(16, 16), view_id=1)
    internal = candidates_from_logits(classes, masks, bounds=(8, 0, 16, 8), **kwargs)
    outer = candidates_from_logits(classes, masks, bounds=(0, 0, 8, 8), **kwargs)
    assert not internal
    # This mask also touches the internal bottom boundary; isolate an outer edge.
    masks[:, 5:, :] = -8.
    outer = candidates_from_logits(classes, masks, bounds=(0, 0, 8, 8), **kwargs)
    assert len(outer) == 1


def test_rejects_nonfinite_probability():
    soft = torch.ones(1, 3, 3)*float('nan')
    with unittest.TestCase().assertRaises(ValueError):
        refine_probability_masks(soft, torch.ones_like(soft, dtype=torch.bool),
                                 torch.ones(3, 3), torch.ones(3, 3, dtype=torch.bool))


if __name__ == '__main__':
    suite = unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in list(globals().items())
                               if name.startswith('test_') and callable(value))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
