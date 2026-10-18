"""Synthetic fixtures test summarization logic only; they are never experiment results."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from tools.summarize_training_methods import (COCO, SEEDS, VARIANTS, development_gate, paired, check_pairing,
                                            audit_loss_trace, describe, audit)


def fixture_rows(gains=None):
    gains = gains or {'bce2': (2., 2., 2.), 'bace': (3., 3., 3.), 'freqfusion': (3., 3., 3.)}
    rows = []
    for seed_index, seed in enumerate(SEEDS):
        for variant in VARIANTS:
            metrics = {k: 10. for k in COCO}
            metrics['mask_ap_small'] += gains[variant][seed_index] if variant != 'baseline' else 0
            metrics['mask_ap'] = 50.
            rows.append({'dataset':'graspnet', 'split':'val', 'seed':seed, 'variant':variant,
                         'metrics':metrics, 'timing':{'mean_ms':20. if variant == 'freqfusion' else 10.}})
    return rows


class SummaryFixtures(unittest.TestCase):
    def test_gate_tie_then_latency(self):
        result = development_gate(fixture_rows(), True)
        self.assertEqual(result['preferred_optional_variant'], 'bace')
        self.assertFalse(result['online_default_changes'])
        self.assertTrue(result['choices']['bce2']['passed'])

    def test_second_tie_break_is_overall_ap(self):
        rows = fixture_rows()
        for row in rows:
            if row['variant'] == 'freqfusion':
                row['metrics']['mask_ap'] += .1
        self.assertEqual(development_gate(rows,True)['preferred_optional_variant'],'freqfusion')

    def test_partial_suppresses_final_gate_even_with_complete_development(self):
        result = development_gate(fixture_rows(),False)
        self.assertEqual(result['status'],'withheld_incomplete_matrix')
        self.assertIsNone(result['preferred_optional_variant'])

    def test_gate_strict_positive_and_ap_guardrail(self):
        rows = fixture_rows({'bce2':(8.,0.,-1.),'bace':(2.,2.,2.),'freqfusion':(3.,3.,3.)})
        for row in rows:
            if row['variant']=='freqfusion':
                row['metrics']['mask_ap'] -= 1.001
        result=development_gate(rows,True)
        self.assertEqual(result['eligible_ranked'],['bace'])
        self.assertFalse(result['choices']['bce2']['conditions']['at_least_two_positive_seeds'])

    def test_test_or_tless_scores_cannot_select_method(self):
        rows=fixture_rows(); expected=development_gate(rows,True)
        extra=copy.deepcopy(rows)
        for row in extra:
            row['split']='test';row['metrics']['mask_ap_small']=100. if row['variant']=='freqfusion' else 0.
        self.assertEqual(expected,development_gate(rows+extra,True))
        for row in extra:
            row['dataset']='tless'
        self.assertEqual(expected,development_gate(rows+extra,True))

    def test_bace_minus_bce2_is_paired_by_seed(self):
        result=paired(list(reversed(fixture_rows())),'bace','bce2','val')
        self.assertEqual([p['seed'] for p in result['pairs']],list(SEEDS))
        self.assertEqual(result['descriptive_delta_pp']['mask_ap_small']['values'],[1.,1.,1.])
        self.assertEqual(result['descriptive_delta_pp']['mask_ap_small']['sample_std_ddof1'],0.)

    def test_sample_standard_deviation_and_nulls(self):
        stats=describe([1.,2.,3.,None])
        self.assertEqual(stats['mean'],2.)
        self.assertEqual(stats['sample_std_ddof1'],1.)
        self.assertEqual(stats['missing'],1)

    def test_mismatched_initial_rng_or_sampling_hash_rejected(self):
        record={'seed':17,'variant':'baseline','sampling_trace_sha256':'same','shared_initialization':{'sha256':'same'},
                'initial_torch_cpu_rng_sha256':'same','initial_torch_cuda_rng_sha256':'same','dataset_sha256':'same',
                'manifest_sha256':'same','fixed_training':{},'base_config':{},'init_checkpoint_sha256':'same'}
        other={**record,'variant':'freqfusion'}
        self.assertTrue(check_pairing([record,other])['17']['checks']['sampling_trace_sha256']['matches'])
        for key in ('initial_torch_cuda_rng_sha256','sampling_trace_sha256','shared_initialization'):
            broken={**other,key:'changed'}
            with self.assertRaisesRegex(ValueError,key):
                check_pairing([record,broken])

    def test_finite_trace_requires_matching_total_and_bce2_control(self):
        with tempfile.TemporaryDirectory() as directory:
            loss_path,trace_path=Path(directory)/'loss.jsonl',Path(directory)/'sampling.jsonl'
            trace={'step':1,'batch_indices':[0,1],'sample_ids':['a','b'],
                   'augmentation_rng_before_sha256':'before','augmentation_rng_after_sha256':'after'}
            row={**trace,'loss':2.,'loss_mask':1.,'loss_mask_aux':1.,'grad_norm':.3,
                 'lr_used':[.00002,.0002],'lr_next':[.00002,.0002],'elapsed_s':1.}
            trace_path.write_text(json.dumps(trace)+'\n');loss_path.write_text(json.dumps(row)+'\n')
            self.assertTrue(audit_loss_trace(loss_path,trace_path,steps=1,variant='bce2')['all_logged_losses_gradients_and_lrs_finite'])
            row['loss_mask_aux']=.5;row['loss']=1.5;loss_path.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError,'BCE2'):
                audit_loss_trace(loss_path,trace_path,steps=1,variant='bce2')
            row['loss']=float('nan');loss_path.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError,'Nonfinite'):
                audit_loss_trace(loss_path,trace_path,steps=1,variant='bce2')

    def test_missing_actual_files_can_be_partial_but_never_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            protocol=Path(__file__).resolve().parents[1]/'docs/training_methods_protocol.md'
            partial=audit(root/'training',root/'evaluation',protocol,allow_partial=True)
            self.assertEqual(partial['counts']['completed_training'],0)
            self.assertEqual(partial['counts']['completed_evaluations'],0)
            self.assertEqual(len(partial['missing_or_incomplete']),40)
            self.assertEqual(partial['status'],'partial_no_final_gate')
            self.assertIsNone(partial['development_gate']['preferred_optional_variant'])
            with self.assertRaisesRegex(ValueError,'incomplete'):
                audit(root/'training',root/'evaluation',protocol)


if __name__=='__main__':
    unittest.main()
