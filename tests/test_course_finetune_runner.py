"""CPU checks for the actual runner helpers; no model/checkpoint/GPU loading."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

_SPEC=importlib.util.spec_from_file_location('course_finetune_runner',Path(__file__).resolve().parents[1]/'tools/run_course_finetune.py')
runner=importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


class TinyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.backbone=nn.Linear(2,2,dtype=torch.float64)
        self.model.head=nn.Linear(2,2,dtype=torch.float64)
        self.model.query_embed=nn.Embedding(3,2,dtype=torch.float64)
        self.model.frozen=nn.Parameter(torch.zeros(2,dtype=torch.float64),requires_grad=False)


class CourseFinetuneRunnerTests(unittest.TestCase):
    def test_warmup_cosine_boundaries_and_monotonic_sections(self):
        # Full 14,432 images, effective batch 8, five complete epochs.
        total=9020
        factors=[runner.lr_factor(step,total) for step in range(total)]
        self.assertAlmostEqual(factors[0],.1036)
        self.assertAlmostEqual(factors[249],1.)
        self.assertAlmostEqual(factors[250],1.)
        self.assertAlmostEqual(factors[-1],.1)
        self.assertTrue(all(a<=b for a,b in zip(factors[:250],factors[1:250])))
        self.assertTrue(all(a>=b for a,b in zip(factors[250:],factors[251:])))
        self.assertTrue(all(.1<=x<=1 for x in factors))

    def test_short_smoke_schedule_is_finite(self):
        for total in (1,2,3,10,40):
            with self.subTest(total=total):
                values=[runner.lr_factor(step,total) for step in range(total)]
                self.assertTrue(all(math.isfinite(x) and .1<=x<=1 for x in values))
                if total>=3:self.assertAlmostEqual(values[-1],.1)

    def test_selection_prioritizes_small_then_overall_and_keeps_exact_ties(self):
        def key(small,overall):return runner.selection_key({'metrics':{'mask_ap_small':small,'mask_ap':overall}})
        self.assertGreater(key(.2,.1),key(.19,.99))
        self.assertGreater(key(.2,.4),key(.2,.3))
        self.assertFalse(key(.2,.4)>key(.2,.4))
        self.assertEqual(key(0.,0.),(0.,0.))

    def test_selection_rejects_unavailable_or_nonfinite_values(self):
        for field in ('mask_ap_small','mask_ap'):
            for invalid in (None,float('nan'),float('inf'),-float('inf')):
                with self.subTest(field=field,invalid=invalid):
                    metrics={'mask_ap_small':.1,'mask_ap':.2};metrics[field]=invalid
                    with self.assertRaises(ValueError):runner.selection_key({'metrics':metrics})

    def test_optimizer_exact_coverage_learning_rates_and_decay_exceptions(self):
        adapter=TinyAdapter()
        opt,summary=runner.make_optimizer(adapter,{'backbone_parameter_prefix':'model.backbone.'},.0002)
        grouped={id(p):group for group in opt.param_groups for p in group['params']}
        all_ids=[id(p) for group in opt.param_groups for p in group['params']]
        expected={id(p) for p in adapter.parameters() if p.requires_grad}
        self.assertEqual(set(all_ids),expected)
        self.assertEqual(len(all_ids),len(expected))
        for name,param in adapter.named_parameters():
            if not param.requires_grad:
                self.assertNotIn(id(param),grouped);continue
            group=grouped[id(param)]
            self.assertAlmostEqual(group['lr'],.00002 if name.startswith('model.backbone.') else .0002)
            self.assertEqual(group['base_lr'],group['lr'])
            self.assertEqual(group['weight_decay'],0. if name.endswith('bias') or 'embed' in name else .05)
        self.assertEqual(sum(row['parameters'] for row in summary),sum(p.numel() for p in adapter.parameters() if p.requires_grad))

    def test_optimizer_real_update_respects_backbone_multiplier(self):
        adapter=TinyAdapter()
        with torch.no_grad():
            for parameter in adapter.parameters():parameter.zero_()
        opt,_=runner.make_optimizer(adapter,{'backbone_parameter_prefix':'model.backbone.'},.0002)
        for parameter in adapter.parameters():
            if parameter.requires_grad:parameter.grad=torch.ones_like(parameter)
        opt.step()
        self.assertAlmostEqual(float(adapter.model.head.weight[0,0]/adapter.model.backbone.weight[0,0]),10.,places=9)

    def test_optimizer_rejects_wrong_backbone_prefix(self):
        with self.assertRaises(ValueError):runner.make_optimizer(TinyAdapter(),{'backbone_parameter_prefix':'missing.'},.0002)

    def test_complete_epoch_order_is_unique_shared_and_epoch_specific(self):
        mr=runner.epoch_order(14432,17,0)
        # Architecture consumes unrelated RNG draws between calls.
        torch.manual_seed(421);torch.randn(73)
        m2=runner.epoch_order(14432,17,0)
        self.assertEqual(mr,m2)
        self.assertEqual(len(mr),14432)
        self.assertEqual(set(mr),set(range(14432)))
        next_epoch=runner.epoch_order(14432,17,1)
        self.assertNotEqual(mr,next_epoch)
        self.assertEqual(set(mr),set(next_epoch))

    @staticmethod
    def state(epoch,images,step,stage='training'):
        return {'epoch':epoch,'images_in_epoch':images,'global_step':step,
                'stage':stage,'val_results':[{'epoch':i+1} for i in range(epoch)]}

    def test_resume_accepts_training_validation_and_finished_boundaries(self):
        valid=[self.state(0,0,0),self.state(0,2400,300),
               self.state(0,14432,1804),self.state(0,14432,1804,'validation'),
               self.state(1,0,1804),self.state(4,14432,9020,'validation'),
               self.state(5,0,9020)]
        for state in valid:
            with self.subTest(state=state):runner.validate_resume_state(state,14432,8,5)

    def test_resume_rejects_wrong_counts_stage_and_history(self):
        invalid=[self.state(0,1,1),self.state(0,2400,301),self.state(0,14433,1805),
                 self.state(0,8,1,'validation'),self.state(0,0,0,'unknown'),
                 self.state(5,8,9021),self.state(5,0,9020,'validation'),
                 self.state(6,0,10824),self.state(0,0,-1)]
        history=self.state(1,0,1804);history['val_results']=[];invalid.append(history)
        fractional=self.state(0,8,1);fractional['images_in_epoch']=8.;invalid.append(fractional)
        boolean=self.state(0,0,0);boolean['global_step']=False;invalid.append(boolean)
        for state in invalid:
            with self.subTest(state=state),self.assertRaises(ValueError):runner.validate_resume_state(state,14432,8,5)

    def test_resume_remaining_batches_have_no_lost_or_duplicate_images(self):
        from torch.utils.data import DataLoader,Subset
        order=runner.epoch_order(14432,17,0)
        cursor=2400
        runner.validate_resume_state(self.state(0,cursor,300),14432,8,5)
        for micro_batch in (8,4):
            with self.subTest(micro_batch=micro_batch):
                remaining=[]
                for batch in DataLoader(Subset(list(range(14432)),order[cursor:]),batch_size=micro_batch,shuffle=False,num_workers=0):
                    remaining.extend(batch.tolist())
                self.assertEqual(order[:cursor]+remaining,order)
                self.assertEqual(len(set(order[:cursor]+remaining)),14432)

    def test_partial_final_accumulation_group_is_valid_step_boundary(self):
        # An 11-image fixture with effective batch 8 retains its last 3 images.
        runner.validate_resume_state(self.state(0,8,1),11,8,1)
        runner.validate_resume_state(self.state(0,11,2,'validation'),11,8,1)
        runner.validate_resume_state(self.state(1,0,2),11,8,1)
        with self.assertRaises(ValueError):runner.validate_resume_state(self.state(0,10,2),11,8,1)

    def test_log_tail_is_archived_and_checkpoint_prefix_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'training_log.jsonl'
            lines=[json.dumps({'global_step':i,'loss':.1*i})+'\n' for i in range(1,6)]
            path.write_text(''.join(lines))
            receipt=runner.archive_rolled_back_log(path,3)
            self.assertEqual(path.read_text(),''.join(lines[:3]))
            self.assertEqual(Path(receipt['path']).read_text(),''.join(lines[3:]))
            self.assertEqual(receipt['rows'],2)
            self.assertIsNone(runner.archive_rolled_back_log(path,3))
            self.assertEqual(len(list(Path(folder).glob('training_log_rolled_back_*.jsonl'))),1)

    def test_missing_log_is_noop_and_malformed_log_fails_without_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'training_log.jsonl'
            self.assertIsNone(runner.archive_rolled_back_log(path,0))
            path.write_text('{"global_step":1}\n{"global_step":')
            before=path.read_bytes()
            with self.assertRaises(ValueError):runner.archive_rolled_back_log(path,1)
            self.assertEqual(path.read_bytes(),before)

    def test_checkpoint_failed_write_does_not_destroy_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'last.pt'
            runner.save_torch(path,{'step':2,'tensor':torch.tensor([4])})
            original=path.read_bytes()
            def failure(value,temporary):
                Path(temporary).write_bytes(b'partial checkpoint')
                raise OSError('simulated disk failure')
            with patch.object(runner.torch,'save',side_effect=failure):
                with self.assertRaises(OSError):runner.save_torch(path,{'step':3})
            self.assertEqual(path.read_bytes(),original)
            self.assertEqual(torch.load(path,weights_only=True)['step'],2)


if __name__=='__main__':unittest.main()
