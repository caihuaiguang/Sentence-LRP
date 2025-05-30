## A Simple Modification of AttnLRP for Attributing a Sentence to the Context

![Sentence-Level Heatmap](https://github.com/caihuaiguang/Sentence-LRP/blob/main/examples/sentence_LRP.png?raw=true)


When explaining a specific sentence output by a large language model, such as:

> *"Neil Armstrong is considered to be the most famous person in the history of the Moon landing. His iconic quote, 'That’s one small step for man, one giant leap for mankind,' became an instant classic and is still remembered today."*

The **AttnLRP** method can successfully trace the relevant supporting context from a source document:

> *The mission was led by astronauts Neil Armstrong, Edwin “Buzz” Aldrin, and Michael Collins. Armstrong’s famous words upon landing were, “That’s one small step for man, one giant leap for mankind.”*

The complete code and result can be seen in [Sentence-level LRP Colab](https://colab.research.google.com/drive/163TSyjS9GeRagDEB-kUe2AsVnQ0K-Ai0?usp=sharing).

And the core implementation is:

```python
# Get predicted logits for the relevant token positions
target_logits = output_logits[0, start_idx-1:end_idx-1]  # shape: [target_len, vocab_size]

# For each position, take the max logit value
max_logits_per_token = torch.max(target_logits, dim=-1).values  # shape: [target_len]

# Sum them
sum_max_logits = max_logits_per_token.sum()

# Backward pass (the relevance is initialized with the value of sum_max_logits)
sum_max_logits.backward()
```

## Original 
<div align="center">
  <img src="docs/source/_static/lxt_logo.png" width="300"/>

  <h3>Layer-wise Relevance Propagation for Transformers</h3>
  <p><i></i></p>

  [![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org)
  [![Read the Docs](https://img.shields.io/badge/-Docs-blue?style=for-the-badge&logo=Read-the-Docs&logoColor=white)](https://lxt.readthedocs.io)
  [![License](https://img.shields.io/badge/License-BSD_3--Clause-green.svg?style=for-the-badge)](https://opensource.org/licenses/BSD-3-Clause)
</div>

## Accelerating eXplainable AI research for LLMs & ViTs

#### 🔥 Highly efficient & Faithful Attributions

Attention-aware LRP (AttnLRP) **outperforms** gradient-, decomposition- and perturbation-based methods, provides faithful attributions for the **entirety** of a black-box transformer model while scaling in computational complexitiy $O(1)$ and memory requirements $O(\sqrt{N})$ with respect to the number of layers.

#### 🔎 Latent Feature Attribution & Visualization
Since we get relevance values for each single neuron in the model as a by-product, we know exactly how important each neuron is for the prediction of the model. Combined with Activation Maximization, we can label neurons or SAE features in LLMs and even steer the generation process of the LLM by activating specialized knowledge neurons in latent space!

#### 📚 Paper
For the mathematical details and foundational work, please take a look at our paper:  
[Achtibat, et al. “AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for Transformers.” ICML 2024.](https://proceedings.mlr.press/v235/achtibat24a.html)  

#### 🏆 Hall of Fame
A collection of papers that have utilized LXT:

- [Arras, et al. “Close Look at Decomposition-based XAI-Methods for Transformer Language Models.” arXiv preprint, 2025.](https://arxiv.org/abs/2502.15886)
- [Pan, et al. “The Hidden Dimensions of LLM Alignment: A Multi-Dimensional Safety Analysis.” arXiv preprint, 2025.](https://arxiv.org/abs/2502.09674)
- [Hu, et al. “LRP4RAG: Detecting Hallucinations in Retrieval-Augmented Generation via Layer-wise Relevance Propagation“ arXiv preprint, 2024.](https://arxiv.org/abs/2408.15533)
- [Sarti, et al. “Quantifying the Plausibility of Context Reliance in Neural Machine Translation.” ICLR 2024.](https://arxiv.org/abs/2310.01188)
[![Demo](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/gsarti/mirage)


#### 📄 License
This project is licensed under the BSD-3 Clause License, which means that LRP is a patented technology that can only be used free of charge for personal and scientific purposes.

## Getting Started 
### 🛠️ Installation 

```bash
pip install lxt
```

Tested with: `transformers==4.48.3`, `torch==2.6.0`, `python==3.11`

### 🚀 Quickstart with 🤗 LLaMA, BERT, GPT2, Qwen, Mixtral & many more
You find example scripts in the `examples/*` directory. For an in-depth tutorial, take a look at the [Quickstart in the Documentation](https://lxt.readthedocs.io/en/latest/quickstart.html).
To get an overview, you can keep reading below ⬇️


## How LXT Works

Layer-wise Relevance Propagation is a rule-based backpropagation algorithm. This means, that we can implement LRP in a single backward pass!
For this, LXT offers two different approaches:

### 1. Efficient Implementation
Uses a Gradient*Input formulation, which simplifies LRP to a standard & fast gradient computation via monkey patching the model class.


```python
from lxt.efficient import monkey_patch

# Patch module first
monkey_patch(your_module)

# Forward pass with gradient tracking
outputs = model(inputs_embeds=input_embeds.requires_grad_())

# Backward pass
outputs.logits[...].backward()

# Get relevance at *ANY LAYER* in your model. Simply multiply the gradient * activation!
# here for the input embeddings:
relevance = (input_embeds.grad * input_embeds).sum(-1)
```
This is the **recommended approach** for most users as it's significantly faster and easier to use. This implementation technique is introduced in [Arras, et al. “Close Look at Decomposition-based XAI-Methods for Transformer Language Models.” arXiv preprint, 2025.](https://arxiv.org/abs/2502.15886)
 
### 2. Mathematical Explicit Implementation
This was used in the original [ICML 2024 paper](https://proceedings.mlr.press/v235/achtibat24a.html). It's more complex and slower, but useful for understanding the mathematical foundations of LRP.


To achieve this, we have implemented [custom PyTorch autograd Functions](https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html) for commonly used operations in transformers. These functions behave identically in the forward pass, but substitute the gradient with LRP attributions in the backward pass. To compute the $\varepsilon$-LRP rule for a linear function $y = W x + b$, you can simply write
```python
import lxt.explicit.functional as lf

y = lf.linear_epsilon(x.requires_grad_(), W, b)
y.backward(y)

relevance = x.grad
```

There are also "super-functions" that wrap an arbitrary nn.Module and compute LRP rules via automatic vector-Jacobian products! These rules are simple to attach to models:

```python
from lxt.explicit.core import Composite
import lxt.explicit.rules as rules

model = nn.Sequential(
  nn.Linear(10, 10),
  RootMeanSquareNorm(),
)

Composite({
  nn.Linear: rules.EpsilonRule,
  RootMeanSquareNorm: rules.IdentityRule,
}).register(model)

print(model)
```
<div align="left">
  <img src="docs/source/_static/terminal.png" width="400"/>
</div>


## Documentaion
[Click here](https://lxt.readthedocs.io) to read the documentation.

## Contribution
Feel free to explore the code and experiment with different datasets and models. We encourage contributions and feedback from the community. We are especially grateful for providing support for new model architectures! 🙏


## Citation
```
@InProceedings{pmlr-v235-achtibat24a,
  title = {{A}ttn{LRP}: Attention-Aware Layer-Wise Relevance Propagation for Transformers},
  author = {Achtibat, Reduan and Hatefi, Sayed Mohammad Vakilzadeh and Dreyer, Maximilian and Jain, Aakriti and Wiegand, Thomas and Lapuschkin, Sebastian and Samek, Wojciech},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  pages = {135--168},
  year = {2024},
  editor = {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = {235},
  series = {Proceedings of Machine Learning Research},
  month = {21--27 Jul},
  publisher = {PMLR}
}
```

## Acknowledgements
The code is heavily inspired by [Zennit](https://github.com/chr5tphr/zennit), a tool for LRP attributions in PyTorch using hooks. Zennit is 100% compatible with the **explicit** version of LXT and offers even more LRP rules 🎉
