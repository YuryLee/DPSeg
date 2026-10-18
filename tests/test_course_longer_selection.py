import unittest

from tools.run_course_longer_point import select_validation
from tools.run_course_longer_tiled import rank_candidates


class LongerSelectionTests(unittest.TestCase):
    def test_primary_aps_is_not_replaced_by_old_ap_drop_gate(self):
        metrics={
            'native_probability_depth':{'mask_ap_small':.50,'mask_ap':.90},
            'point':{'mask_ap_small':.60,'mask_ap':.80},
            'point_probability_depth':{'mask_ap_small':.59,'mask_ap':.89},
            'point_depth':{'mask_ap_small':.55,'mask_ap':.90},
            'point_depth_probability_depth':{'mask_ap_small':.57,'mask_ap':.90}}
        choice=select_validation([{'epoch':1,'metrics':metrics,'checkpoint':{'epoch':1}}])
        self.assertEqual(choice['selected']['variant'],'point')
        self.assertFalse(choice['selected']['eligible'])

    def test_whole_image_can_win_and_historical_tie_is_retained(self):
        historical={'old':{'mask_ap_small':.62,'mask_ap':.85}}
        current={'whole':{'mask_ap_small':.64,'mask_ap':.84},
                 'tile':{'mask_ap_small':.63,'mask_ap':.90}}
        self.assertEqual(rank_candidates(current,historical)[0]['variant'],'whole')
        current={'tile':{'mask_ap_small':.62,'mask_ap':.85}}
        self.assertEqual(rank_candidates(current,historical)[0]['origin'],'historical')
        with self.assertRaises(ValueError):
            rank_candidates({'bad':{'mask_ap_small':None,'mask_ap':.8}},historical)
        with self.assertRaises(ValueError):
            rank_candidates({'bad':{'mask_ap_small':float('nan'),'mask_ap':.8}},historical)


if __name__=='__main__':unittest.main()
