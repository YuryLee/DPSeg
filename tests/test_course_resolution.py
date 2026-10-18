import json
from pathlib import Path
import tempfile
import unittest

import torch

from rgbd_seg.course_resolution import heterogeneous_rank_order, local_num_masks, rank_batches
from rgbd_seg.course_distributed_eval import merge_shards


class ResolutionTests(unittest.TestCase):
    def test_larger_batches_include_exact_tail_without_duplicates(self):
        order = torch.randperm(14432, generator=torch.Generator().manual_seed(17)).tolist()
        batches = [6,6,6,6,3,3,3]
        ranks = [rank_batches(order, rank=r, batch_sizes=batches) for r in range(7)]
        self.assertEqual({len(r) for r in ranks}, {438})
        self.assertEqual([len(r[-1]) for r in ranks], [2,2,2,2,1,1,1])
        recovered = [index for step in range(438) for rank in ranks for index in rank[step]]
        self.assertEqual(recovered, order)
        self.assertEqual(len(set(recovered)), 14432)
        with self.assertRaises(ValueError):
            rank_batches(order[:34], rank=0, batch_sizes=batches)

    def test_all_course_images_once_with_heterogeneous_gpus(self):
        order = torch.randperm(14432, generator=torch.Generator().manual_seed(17)).tolist()
        batches = [4, 4, 4, 4, 2, 2, 2]
        ranks = [heterogeneous_rank_order(order, rank=r, batch_sizes=batches) for r in range(7)]
        recovered = []
        for step in range(14432//sum(batches)):
            for values, batch in zip(ranks, batches):
                recovered.extend(values[step*batch:(step+1)*batch])
        self.assertEqual(recovered, order)
        self.assertEqual(len(set(sum(ranks, []))), 14432)
        with self.assertRaises(ValueError):
            heterogeneous_rank_order(order[:-1], rank=0, batch_sizes=batches)

    def test_weighted_rank_gradient_matches_serial_microbatch_objective(self):
        batches = [4, 4, 4, 4, 2, 2, 2]
        accumulation = 2
        world, effective = len(batches), sum(batches)*accumulation
        ranks = []
        reference = torch.tensor([.3, -.5], dtype=torch.float64, requires_grad=True)
        reference_loss = 0
        for rank, batch in enumerate(batches):
            parameter = reference.detach().clone().requires_grad_()
            local_loss = 0
            for micro in range(accumulation):
                data = torch.arange(batch*2, dtype=torch.float64).reshape(batch, 2)+rank+micro
                target = data[:, 0]*.2
                local_loss += ((data@parameter-target)**2).mean()*world*batch/effective
                reference_loss += ((data@reference-target)**2).mean()*batch/effective
            ranks.append(torch.autograd.grad(local_loss, parameter)[0])
        expected = torch.autograd.grad(reference_loss, reference)[0]
        torch.testing.assert_close(torch.stack(ranks).mean(0), expected)

    def test_native_mask_normalization_stays_local(self):
        self.assertEqual(local_num_masks(None, [torch.zeros(3), torch.zeros(7)], 'cpu').item(), 10)
        self.assertEqual(local_num_masks(None, [torch.empty(0)], 'cpu').item(), 1)

    def test_evaluation_merge_rejects_duplicate_missing_and_foreign_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root/'a.jsonl', root/'b.jsonl']
            def row(key):
                return json.dumps({'image_id': key, 'predictions': []})+'\n'
            paths[0].write_text(row(3)+row(1))
            paths[1].write_text(row(2))
            self.assertEqual(merge_shards(paths, [1, 2, 3], root/'out.jsonl'), 0)
            self.assertEqual([json.loads(s)['image_id'] for s in (root/'out.jsonl').read_text().splitlines()], [1,2,3])
            for value in (row(1), '', row(4)):
                paths[1].write_text(value)
                with self.assertRaises(ValueError):
                    merge_shards(paths, [1, 2, 3], root/'invalid.jsonl')

    def test_initial_weights_and_small_mask_preprocessing(self):
        from rgbd_seg.course_dafusion import build_trainable as original
        from rgbd_seg.course_resolution import build_trainable
        root = Path(__file__).resolve().parents[1]
        checkpoint = root/'pretrained/mask2former_swin_tiny_coco_instance'
        torch.set_num_threads(2)
        base, _ = original(checkpoint, device='cpu', seed=17)
        high, metadata = build_trainable(checkpoint, device='cpu', seed=17, resolution=1024)
        for name, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, high.state_dict()[name]), name)
        encoder = high.model.model.pixel_level_module.encoder
        for stage, fusion in enumerate(encoder.fusions):
            modules = [m for m in fusion.modules() if m.__class__.__name__ == 'ContinuousPositionBias']
            self.assertEqual(len(modules), 4)
            self.assertEqual({m.max_pairs for m in modules}, {0 if stage < 2 else 8000000})
        self.assertEqual(metadata['fusion']['rpb_active_at_input'], [False, False, True, True])
        sample = {'image': torch.zeros(3,1024,1024), 'depth_m': torch.ones(1,1024,1024),
                  'depth_valid': torch.ones(1,1024,1024,dtype=torch.bool),
                  'camera_K': torch.tensor([[1000.,0,512],[0,1000,512],[0,0,1]])}
        masks = torch.zeros(1,1024,1024,dtype=torch.uint8)
        masks[0,1023,1023] = 1
        targets = [{'masks': masks, 'labels': torch.ones(1,dtype=torch.int64)}]
        _, low_info = base.prepare_batch([sample], targets)
        batch, high_info = high.prepare_batch([sample], targets)
        self.assertEqual(low_info['resized_empty_instances'], [1])
        self.assertEqual(high_info['resized_empty_instances'], [0])
        self.assertEqual(tuple(batch['pixel_values'].shape), (1,7,1024,1024))
        self.assertEqual(int(batch['mask_labels'][0].sum()), 1)


if __name__ == '__main__':
    unittest.main()
