import tempfile
import unittest
from pathlib import Path
from tools.write_training_methods_report import parse_tests_log,verify_final_summary


class ReportChecks(unittest.TestCase):
    def test_test_count_comes_from_successful_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'test.log'
            path.write_text('test_example ... ok\n\nRan 114 tests in 7.217s\n\nOK\n')
            self.assertEqual(parse_tests_log(path)['count'],114)
            path.write_text('Ran 115 tests in 8.5s\n\nFAILED (failures=1)\n')
            with self.assertRaisesRegex(ValueError,'successful'):
                parse_tests_log(path)

    def test_partial_report_is_rejected_before_output(self):
        with self.assertRaisesRegex(ValueError,'partial'):
            verify_final_summary({'status':'partial_no_final_gate'})


if __name__=='__main__':
    unittest.main()
