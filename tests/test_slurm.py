import unittest
from unittest.mock import patch

from frust.stepper import Stepper
from frust.utils.slurm import detect_job_id


class JobIdDetectionTests(unittest.TestCase):
    def test_explicit_job_id_wins(self):
        self.assertEqual(detect_job_id(42, live=True), 42)

    def test_slurm_job_id_is_detected(self):
        with patch.dict("os.environ", {"SLURM_JOB_ID": "123456"}, clear=True):
            self.assertEqual(detect_job_id(None, live=True), 123456)

    def test_non_live_local_run_has_no_job_id(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(detect_job_id(None, live=False))

    def test_stepper_logger_uses_local_without_job_id(self):
        with patch("frust.stepper.detect_job_id", return_value=None):
            step = Stepper(debug=True, save_output_dir=False)

        self.assertTrue(step.logger.name.endswith(".local"))

    def test_stepper_logger_uses_explicit_job_id(self):
        step = Stepper(job_id=42, debug=True, save_output_dir=False)

        self.assertTrue(step.logger.name.endswith(".job42"))


if __name__ == "__main__":
    unittest.main()
