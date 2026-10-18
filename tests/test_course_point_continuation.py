import copy
import unittest
from unittest.mock import patch

from tools.run_course_point_continuation import selection_history, select_validation
from tools.run_course_point_refinement import VARIANTS


def result(epoch, aps):
    return {'epoch':epoch,'checkpoint':{'epoch':epoch,'path':f'head_{epoch}.pt'},
            'metrics':{variant:{'mask_ap_small':aps,'mask_ap':.8} for variant in VARIANTS}}


class ContinuationSelectionTests(unittest.TestCase):
    def test_parent_is_retained_when_more_training_does_not_improve_aps(self):
        parent=result(2,.54)
        with patch('tools.run_course_point_continuation.parent_inputs',return_value=({},parent)):
            rows=selection_history({'scope':'full_experiment'},[result(3,.52),result(4,.53)])
        self.assertEqual(select_validation(rows)['selected']['epoch'],2)
        improved=copy.deepcopy(rows)
        improved[-1]['metrics']['point_depth_probability_depth']['mask_ap_small']=.55
        chosen=select_validation(improved)['selected']
        self.assertEqual((chosen['epoch'],chosen['variant']),(4,'point_depth_probability_depth'))

    def test_engineering_subset_is_not_ranked_against_full_validation(self):
        with patch('tools.run_course_point_continuation.parent_inputs',return_value=({},result(2,.54))):
            rows=selection_history({'scope':'engineering_smoke'},[result(3,.10)])
        self.assertEqual([row['epoch'] for row in rows],[3])


if __name__=='__main__':unittest.main()
