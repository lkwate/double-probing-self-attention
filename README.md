# Double Probing Self Attention For Natural Language Inference

The vanilla self-attention mechanism introduction by [Vaswani et al.](https://arxiv.org/abs/1706.03762) could be viewed as a mixture of a probing operation and a retrieval one. In this project, we turn the Natural Language Inference task into a task consisting in probing the semantic content of the premise with the representation of the hypothesis and vice-versa. This double probing could be observed as a dual cross attention between the premise and hypothesis.
We use [Poetry](https://python-poetry.org/) as the packaging and dependency manager

# Setting Up
Follow the instructions in the original documentation for the installation [Poetry's documentation](https://python-poetry.org/)
```sh
git clone https://github.com/lkwate/double-probing-self-attention.git repo 
cd repo
poetry init
poetry install
```

# Training
```sh
bash launcher.sh
```