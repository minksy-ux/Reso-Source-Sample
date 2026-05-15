import tempfile
import unittest
from pathlib import Path

import numpy as np

from reso_demo import IsingModel, analyze, create_model_from_reso


class TestResoDemo(unittest.TestCase):
    def test_ising_model_sample_shape(self):
        model = IsingModel("/models/test", size=16, J=-1.0)
        samples = model.sample_states(50)
        self.assertEqual(samples.shape, (50, 16))
        self.assertTrue(np.all(np.isin(samples, [-1, 1])))

    def test_create_model_from_reso(self):
        meta = {"type": "ising", "path": "/models/test", "size": 25, "J": -0.5}
        model = create_model_from_reso(meta)
        self.assertEqual(model.size, 25)
        self.assertEqual(model.J, -0.5)
        self.assertEqual(model.path, "/models/test")

    def test_analyze_writes_stats(self):
        output_dir = Path(tempfile.mkdtemp())
        samples = np.random.choice([-1, 1], size=(10, 16))
        analyze(output_dir)(samples)
        stats_file = output_dir / "ising_sample_stats.txt"
        self.assertTrue(stats_file.exists())
        content = stats_file.read_text()
        self.assertIn("Sample summary:", content)
        self.assertIn("- steps: 10", content)


if __name__ == "__main__":
    unittest.main()
