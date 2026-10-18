"""Launch preconditions: only audited independent arms may start."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.launch_course_crop_focal_ddp import load_protocols, sha


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'source.py'
        self.source.write_text('# synthetic launch-test fixture, never executed\n')
        source_hashes = {str(self.source):sha(self.source)}
        parent = self.save('parent.json',{'status':'fixture'})
        decision = self.save('decision.json',{'status':'completed','selected_resolution':1024,
                                            'protocol_sha256':parent['sha256']})
        engineering = self.save('engineering.json',{'status':'passed','scope':'seven_gpu_crop_focal_smoke',
            'source_sha256':source_hashes,'resolution':1024,'world_size':7,
            'per_rank_batch_sizes':[6,6,6,6,3,3,3],
            'methods':{'crop':{'status':'passed'},'focal':{'status':'passed'}}})
        base = dict(status='frozen',resolution_selection=decision,engineering_validation=engineering,
            parent_resolution_protocol=parent,selected_resolution=1024,source_sha256=source_hashes,
            hardware={'physical_gpus':['0','1','2','3','5','6','7']},archive='fixture',checkpoint='fixture',
            input_sha256={},evaluation={},training={'epochs':5,'per_rank_batch_sizes':[6,6,6,6,3,3,3]})
        self.paths = []
        for method in ('crop','focal'):
            p = copy.deepcopy(base);p['intervention']={'method':method}
            p['training']['augmentation']=method
            self.paths.append(Path(self.save(method+'.json',p)['path']))

    def save(self,name,value):
        path = self.root/name;path.write_text(json.dumps(value))
        return {'path':str(path),'sha256':sha(path)}

    def modify(self,index,change):
        path = self.paths[index];p=json.loads(path.read_text());change(p);path.write_text(json.dumps(p))

    def test_allows_only_augmentation_difference_in_training(self):
        self.assertEqual(len(load_protocols(self.paths)),2)
        self.modify(1,lambda p:p['training'].update(epochs=6))
        with self.assertRaisesRegex(ValueError,'training budgets'):
            load_protocols(self.paths)

    def test_rejects_drafts_and_wrong_order(self):
        with self.assertRaisesRegex(ValueError,'crop, then focal'):
            load_protocols(self.paths[::-1])
        self.modify(0,lambda p:p.update(status='draft'))
        with self.assertRaisesRegex(ValueError,'frozen protocols'):
            load_protocols(self.paths)

    def test_rejects_changed_runtime_sources(self):
        self.source.write_text('# changed fixture\n')
        with self.assertRaisesRegex(ValueError,'Frozen source changed'):
            load_protocols(self.paths)

    def test_cpu_pass_cannot_substitute_for_gpu_smoke(self):
        cpu_receipt = self.save('cpu_only.json',{'status':'passed','scope':'CPU only'})
        self.modify(0,lambda p:p.update(engineering_validation=cpu_receipt))
        with self.assertRaisesRegex(ValueError,'GPU engineering'):
            load_protocols(self.paths)

    def test_rejects_wrong_resolution(self):
        self.modify(1,lambda p:p.update(selected_resolution=800))
        with self.assertRaisesRegex(ValueError,'resolution selection'):
            load_protocols(self.paths)


if __name__ == '__main__':
    unittest.main()
