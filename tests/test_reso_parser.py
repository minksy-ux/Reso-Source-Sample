import tempfile
import unittest

from reso_parser import parse_reso_file


class TestResoParser(unittest.TestCase):
    def test_parse_simple_model(self):
        content = """model IsingModel {
path: \"/models/ising2d\"
nodes: array<PNode> = grid(10, 10)
factors coupling {
  J = -1.0
}
}"""
        with tempfile.NamedTemporaryFile("w+", suffix=".reso", delete=False) as handle:
            handle.write(content)
            handle.flush()
            path = handle.name

        parsed = parse_reso_file(path)
        self.assertEqual(parsed["name"], "IsingModel")
        self.assertEqual(parsed["path"], "/models/ising2d")
        self.assertEqual(parsed["J"], -1.0)
        self.assertEqual(parsed["size"], 100)
        self.assertEqual(parsed["type"], "ising")


if __name__ == "__main__":
    unittest.main()
