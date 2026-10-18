import math
import unittest

import torch

from rgbd_seg.course_longer_training import continuation_lr
from rgbd_seg.course_resolution import rank_batches


class LongerTrainingTests(unittest.TestCase):
    def test_five_gpu_accumulation_covers_every_image_and_tail_once(self):
        order = torch.randperm(14432, generator=torch.Generator().manual_seed(17)).tolist()
        batches = [3, 2, 2, 2, 2]
        shards = [rank_batches(order, rank=r, batch_sizes=batches) for r in range(5)]
        recovered = [value for micro in range(len(shards[0]))
                     for shard in shards for value in shard[micro]]
        self.assertEqual(recovered, order)
        self.assertEqual(len(set(recovered)), 14432)
        self.assertEqual(math.ceil(len(shards[0])/3), 438)
        self.assertEqual(sum(len(s[-1]) for s in shards), 11)

    def test_accumulated_ddp_gradient_matches_declared_image_weighted_objective(self):
        for micro_count in (1, 3):
            batches = [3, 2, 2, 2, 2]
            total = sum(batches)*micro_count
            reference = torch.tensor([.3, -.5], dtype=torch.float64, requires_grad=True)
            objective = 0
            gradients = []
            for rank, batch in enumerate(batches):
                parameter = reference.detach().clone().requires_grad_()
                loss = 0
                for micro in range(micro_count):
                    data = torch.arange(batch*2, dtype=torch.float64).reshape(batch, 2)+rank+micro
                    target = data[:, 0]*.2
                    loss += ((data@parameter-target)**2).mean()*len(batches)*batch/total
                    objective += ((data@reference-target)**2).mean()*batch/total
                gradients.append(torch.autograd.grad(loss, parameter)[0])
            torch.testing.assert_close(torch.stack(gradients).mean(0),
                                       torch.autograd.grad(objective, reference)[0])

    def test_learning_rate_start_peak_and_end(self):
        for peak, warmup in [(2e-5, 0), (5e-5, 100)]:
            kwargs = dict(peak_lr=peak, start_lr=2e-5, warmup_steps=warmup, minimum_ratio=.1)
            values = [continuation_lr(i, 4380, **kwargs) for i in range(4380)]
            self.assertAlmostEqual(values[0], 2e-5)
            self.assertAlmostEqual(max(values), peak)
            self.assertAlmostEqual(values[-1], peak*.1)
            self.assertTrue(all(a >= b for a,b in zip(values[warmup:], values[warmup+1:])))


if __name__ == '__main__':
    unittest.main()
