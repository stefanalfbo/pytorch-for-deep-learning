"""Check that the installed examples import safely and run from the project root."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "pytorch_for_deep_learning.neural_network_basics"
EXAMPLES = {
    "activation-functions": (
        "activation_functions",
        ("Comparison of activation functions", "Leaky ReLU", "GELU"),
    ),
    "neuron": (
        "neuron",
        ("neuron1", "neuron2", "Predicted delivery time:"),
    ),
    "simple-neural-network": (
        "simple_neural_network",
        (
            "Epoch 500:",
            "Prediction for a 7.0-mile delivery:",
            "Loss on new, combined data:",
        ),
    ),
    "non-linear-patterns": (
        "non_linear_patterns_with_activation_functions",
        ("Training Complete.", "Final Loss:", "Prediction for a 5.1-mile delivery:"),
    ),
}


class ExampleTests(unittest.TestCase):
    def run_example(self, *args):
        result = subprocess.run(
            args,
            cwd=PROJECT_ROOT,
            env={**os.environ, "MPLBACKEND": "Agg"},
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_imports_do_not_run_examples(self):
        module_names = [f"{PACKAGE}.{module}" for module, _ in EXAMPLES.values()]
        code = (
            "import importlib\n"
            "import matplotlib.pyplot as plt\n"
            "import torch\n"
            "initial_rng = torch.get_rng_state().clone()\n"
            f"for name in {module_names!r}:\n"
            "    module = importlib.import_module(name)\n"
            "    assert callable(module.main), name\n"
            "assert torch.equal(initial_rng, torch.get_rng_state())\n"
            "assert not plt.get_fignums()\n"
        )
        self.assertEqual(self.run_example(sys.executable, "-c", code), "")

    def test_console_commands_run_from_root(self):
        for command, (_, expected_messages) in EXAMPLES.items():
            with self.subTest(command=command):
                executable = shutil.which(command, path=sysconfig.get_path("scripts"))
                self.assertIsNotNone(executable, f"Missing command: {command}")
                output = self.run_example(executable)
                for message in expected_messages:
                    self.assertIn(message, output)

    def test_python_module_runs_from_root(self):
        output = self.run_example(
            sys.executable, "-m", f"{PACKAGE}.activation_functions"
        )
        self.assertIn("Comparison of activation functions", output)


if __name__ == "__main__":
    unittest.main()
