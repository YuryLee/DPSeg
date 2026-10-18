import json
from pathlib import Path
import tempfile
import unittest

from rgbd_seg.course_optimized import file_sha
from tools.build_course_longer_extension_protocol import checked_review


def save(path, value):
    path.write_text(json.dumps(value))
    return {'path': str(path), 'sha256': file_sha(path)}


def fixture(folder):
    entries = []
    # The higher AP belongs to the lower APs arm. Selection must prioritize APs.
    for name, ap, aps in [('lr5e5', .80, .51), ('lr2e5', .85, .50)]:
        sub = folder/name
        sub.mkdir()
        protocol = {'arm': name, 'status': 'frozen', 'selected_resolution': 1024,
                    'input_sha256': {'archive': 'same-data', 'pretrained': 'same-init'},
                    'continuation': {'parent_epoch': 5},
                    'training': {'epochs': 2, 'updates_per_epoch': 438}}
        p = save(sub/'protocol.json', protocol)
        run = {'status': 'completed', 'scope': 'full_train_val', 'test_used': False,
               'protocol_sha256': p['sha256'], 'verified_inputs': protocol['input_sha256'],
               'best_epoch': 7, 'best_validation_key': [aps, ap],
               'parent_validation': {'epoch': 5, 'metrics': {'mask_ap_small': .46, 'mask_ap': .79}}}
        r = save(sub/'run.json', run)
        audit = {'status': 'passed', 'scope': 'completed_base_arm', 'arm': name, 'test_used': False,
                 'protocol_sha256': p['sha256'],
                 'epochs': [{'epoch': epoch, 'AP': ap, 'APs': aps - (7-epoch)*.01,
                             'updates': 438, 'training_images': 14432, 'validation_images': 2584}
                            for epoch in (6, 7)]}
        a = save(sub/'audit.json', audit)
        entries.append({'arm': name, 'protocol': p, 'run': r, 'audit': a})
    review = {'status': 'extension_required_after_convergence_review', 'selection_split': 'val',
              'test_used': False, 'reason': 'Fixture endpoint remains the best candidate',
              'selected_arm': 'lr5e5', 'completed_arms': entries}
    path = folder/'review.json'
    save(path, review)
    return path, review


def stopped_fixture(folder):
    path, review = fixture(folder)
    entry = review['completed_arms'].pop()
    p = json.loads(Path(entry['protocol']['path']).read_text())
    p['training']['epochs'] = 10
    entry['protocol'] = save(Path(entry['protocol']['path']), p)
    run = json.loads(Path(entry['run']['path']).read_text())
    run.update(status='training', protocol_sha256=entry['protocol']['sha256'])
    entry['run'] = save(Path(entry['run']['path']), run)
    full = json.loads(Path(entry['audit']['path']).read_text())
    selection = {'test_used': False, 'parent_candidate': run['parent_validation'],
                 'best_epoch': run['best_epoch'], 'best_key': run['best_validation_key'],
                 'epochs': [{'epoch': v['epoch'], 'metrics': {'mask_ap': v['AP'], 'mask_ap_small': v['APs']}}
                            for v in full['epochs']]}
    entry['selection'] = save(folder/'stopped_selection.json', selection)
    entry['audits'] = []
    for row in full['epochs']:
        audit = {**full, 'scope': 'completed_epoch_only', 'epochs': [row],
                 'protocol_sha256': entry['protocol']['sha256']}
        entry['audits'].append(save(folder/f"stopped_epoch{row['epoch']}.json", audit))
    entry.pop('audit')
    control = {'status': 'failed', 'active_process_alive': False,
               'trials': [{'arm': 'lr2e5', 'pid': 123, 'status': 'failed', 'exit_code': 1,
                           'protocol_sha256': entry['protocol']['sha256']}]}
    stop = {'status': 'stopped_by_user', 'requested_by': 'user', 'arm': 'lr2e5',
            'process_group_absent': True, 'worker_processes_absent': True,
            'user_objective': 'Fixture: stop low-LR training', 'pid': 123,
            'supervisor_exit_code': 1, 'completed_audited_epochs': [6, 7],
            'supervisor_record': save(folder/'stopped_control.json', control)}
    entry['stop'] = save(folder/'stop.json', stop)
    review['stopped_arms'] = [entry]
    save(path, review)
    return path, review


class ExtensionReviewTests(unittest.TestCase):
    def test_APs_winner_is_required_even_when_other_arm_has_higher_AP(self):
        with tempfile.TemporaryDirectory() as directory:
            path, review = fixture(Path(directory))
            selected, evidence = checked_review(path)
            self.assertEqual(selected['arm'], 'lr5e5')
            self.assertEqual(len(evidence), 7)
            review['selected_arm'] = 'lr2e5'
            save(path, review)
            with self.assertRaisesRegex(ValueError, 'not the full-validation winner'):
                checked_review(path)

    def test_both_arms_must_be_present_and_actually_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            path, review = fixture(Path(directory))
            original = json.loads(path.read_text())
            review['completed_arms'] = review['completed_arms'][:1]
            save(path, review)
            with self.assertRaisesRegex(ValueError, 'Both original learning-rate arms'):
                checked_review(path)
            review = original
            entry = review['completed_arms'][1]
            run_path = Path(entry['run']['path'])
            run = json.loads(run_path.read_text())
            run['status'] = 'running'
            entry['run'] = save(run_path, run)
            save(path, review)
            with self.assertRaisesRegex(ValueError, 'training is incomplete'):
                checked_review(path)

    def test_mutated_evidence_or_partial_validation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, review = fixture(Path(directory))
            entry = review['completed_arms'][0]
            audit_path = Path(entry['audit']['path'])
            audit = json.loads(audit_path.read_text())
            audit['epochs'][0]['validation_images'] = 8
            save(audit_path, audit)
            with self.assertRaisesRegex(ValueError, 'evidence changed'):
                checked_review(path)
            entry['audit']['sha256'] = file_sha(audit_path)
            save(path, review)
            with self.assertRaisesRegex(ValueError, 'incomplete training or validation coverage'):
                checked_review(path)

    def test_earlier_best_cannot_use_later_optimizer_as_if_it_matched(self):
        with tempfile.TemporaryDirectory() as directory:
            path, review = fixture(Path(directory))
            entry = review['completed_arms'][0]
            audit_path, run_path = Path(entry['audit']['path']), Path(entry['run']['path'])
            audit, run = json.loads(audit_path.read_text()), json.loads(run_path.read_text())
            audit['epochs'][0]['APs'] = .52
            run.update(best_epoch=6, best_validation_key=[.52, .80])
            entry['audit'], entry['run'] = save(audit_path, audit), save(run_path, run)
            save(path, review)
            with self.assertRaisesRegex(ValueError, 'not a final-epoch checkpoint'):
                checked_review(path)

    def test_explicit_stop_needs_verified_exit_and_all_completed_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            path, review = stopped_fixture(Path(directory))
            selected, _ = checked_review(path)
            self.assertEqual(selected['arm'], 'lr5e5')
            entry = review['stopped_arms'][0]
            stop_path = Path(entry['stop']['path'])
            stop = json.loads(stop_path.read_text())
            stop['process_group_absent'] = False
            entry['stop'] = save(stop_path, stop)
            save(path, review)
            with self.assertRaisesRegex(ValueError, 'verified explicit user stop'):
                checked_review(path)
            stop['process_group_absent'] = True
            entry['stop'] = save(stop_path, stop)
            entry['audits'].pop()
            save(path, review)
            with self.assertRaisesRegex(ValueError, 'completed epoch coverage differs'):
                checked_review(path)

    def test_stopped_arm_scores_cannot_be_discarded_or_restart_authorized(self):
        with tempfile.TemporaryDirectory() as directory:
            path, review = stopped_fixture(Path(directory))
            entry = review['stopped_arms'][0]
            audit_path = Path(entry['audits'][-1]['path'])
            audit = json.loads(audit_path.read_text())
            audit['epochs'][0]['APs'] = .60
            entry['audits'][-1] = save(audit_path, audit)
            selection_path = Path(entry['selection']['path'])
            selection = json.loads(selection_path.read_text())
            selection['epochs'][-1]['metrics']['mask_ap_small'] = .60
            selection['best_key'][0] = .60
            entry['selection'] = save(selection_path, selection)
            save(path, review)
            with self.assertRaisesRegex(ValueError, 'not the full-validation winner'):
                checked_review(path)
            review['selected_arm'] = 'lr2e5'
            save(path, review)
            with self.assertRaisesRegex(ValueError, 'Do not restart'):
                checked_review(path)


if __name__ == '__main__':
    unittest.main()
