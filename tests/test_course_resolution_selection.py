import tempfile
from pathlib import Path
import json
import unittest

from tools.select_course_resolution import metric_key, select_resolution, audit_trial


class SelectionTests(unittest.TestCase):
    def test_requires_small_gain_and_overall_guard_together(self):
        gate = {'APsmall_gain_at_least':.005,'AP_drop_at_most':.005}
        base = {'mask_ap_small':.3,'mask_ap':.7}
        for small,overall,expected in ((.32,.7,1024),(.3,.75,800),(.32,.68,800),(.29,.69,800)):
            actual = select_resolution({'mask_ap_small':small,'mask_ap':overall},base,gate)
            self.assertEqual(actual['selected_resolution'],expected)

    def test_rejects_invalid_selection_metrics(self):
        for value in (None,float('nan'),float('inf'),-1.,1.01,True):
            with self.assertRaises(ValueError):
                metric_key({'mask_ap_small':value,'mask_ap':.7})

    def test_rejects_live_runs_and_completed_smokes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for status,scope in [('training','full_train_val'),('validating','full_train_val'),('completed','engineering_smoke')]:
                (folder/'run.json').write_text(json.dumps({'status':status,'scope':scope}))
                with self.assertRaisesRegex(ValueError,'incomplete or smoke-only'):
                    audit_trial(folder,{'training':{}},'unused',1024)


if __name__ == '__main__':
    unittest.main()
