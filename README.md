# PyTorch for Deep Learning

This repository is for the specialization [PyTorch for Deep Learning](https://learn.deeplearning.ai/specializations/pytorch-for-deep-learning-professional-certificate/information) from [DeepLearning.AI](https://learn.deeplearning.ai/).

## About this Professional Certificate

Building practical deep learning systems means going beyond theory. The **PyTorch for Deep Learning Professional Certificate** teaches you to build and train the deep learning models that power real AI applications, using PyTorch — one of the most widely adopted frameworks in research and industry — to design efficient, reliable systems.

In this 3-course professional certificate, you’ll learn through hands-on projects that mirror the challenges faced by deep learning engineers: designing efficient architectures, applying transfer learning and fine-tuning to pretrained models, using interpretability techniques to understand model behavior, and preparing optimized, portable models with ONNX and experiment tracking tools like MLflow. Along the way, you’ll **gain experience with techniques used across modern AI applications, including pruning and quantization**.

Whether you’re strengthening your career in machine learning, expanding into applied AI, or building your own projects, this certificate gives you the skills and the confidence to turn ideas into working PyTorch models.

## Project structure

```text
src/
└── pytorch_for_deep_learning/
    ├── __init__.py
    ├── __main__.py
    ├── neural_network_basics/
    │   ├── __init__.py
    │   ├── activation_functions.py
    │   ├── neuron.py
    │   ├── simple_neural_network.py
    │   └── non_linear_patterns_with_activation_functions.py
    └── utils/
        ├── __init__.py
        └── helper_utils.py
tests/
└── test_examples.py
```

The `neural_network_basics` package groups the introductory lessons. Shared
plotting helpers live in `utils`. Each example has a `main()` function so that
importing it does not start training or open plots.

## How to run the code

Run these commands from the repository root. `uv run` installs the local package
and its dependencies automatically, including the commands defined in
`pyproject.toml`.

```bash
uv run activation-functions
uv run neuron
uv run simple-neural-network
uv run non-linear-patterns
```

The old root-level script paths have moved. You can also use Python's module
syntax from the root, for example:

```bash
uv run python -m pytorch_for_deep_learning.neural_network_basics.neuron
```

The two network labs display Matplotlib plots; close each plot window to continue.

### Run tests

The tests use Python's built-in `unittest` runner and a non-interactive plotting
backend, so no plot windows open during testing.

```bash
uv run python -m unittest discover -s tests -v
```

### Use ruff

```bash
uv run ruff check
uv run ruff format
```
