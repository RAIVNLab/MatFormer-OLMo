# MatFormer-OLMo

This is a public reproduction and open source release of the [MatFormer](https://arxiv.org/abs/2310.07707)'s language modeling experiments (MatLM). 

MatFormer-OLMo is built on an older codebase of [AI2's OLMo](https://allenai.org/olmo) and also releases 3 model checkpoints trained on the compute cluster of the [Kempner Institute](https://www.harvard.edu/kempner-institute/about/#compute) at the Harvard University.

A similar open source release for the vision encoder experiments of MatFormer (MatViT) can be found in the [MatViT project](https://github.com/google-research/scenic/tree/main/scenic/projects/matvit) of the Scenic library.

## Setup

After cloning this repository, first install the latest [PyTorch](https://pytorch.org) according the official instructions relevant to your environment. Then install the remaining dependencies and code base by running:

```
pip install -e .[dev]
```

Setup the paths appropriately in the [scripts/env_pile_test.sh](./scripts/env_pile_test.sh) for `HF_HOME`; 
[scripts/pile_test.sh](./scripts/pile_test.sh) for the SLURM setup;
[configs/pile-tiny.yaml](./configs/pile-tiny.yaml) for data and save paths.

All the models are trained on the [Pile corpus](https://pile.eleuther.ai/) tokenized using `EleutherAI/gpt-neox-20b` tokenizer. 

## Running LM pre-training jobs

Our training script is [scripts/train.py](./scripts/train.py), which should be launched either through `torchrun` or Slurm (see below) since it only supports distributed training (on GPUs).
The first argument to the training script is a path to a [training configuration file](./configs/).
Then it takes any number of optional arguments that can be used to override values from the configuration file using dot notation.
For example, to change the learning rate you'd pass `--optimizer.learning_rate=0.0001`.

To use MatFormer structure use the `--matformer_factor` flag. Setting `--matformer_factor=1` results in vanilla baseline model, while using `--matformer_factor=8` has 4 exponential granularities in the MLP `{h, h/2, h/4, h/8}`.

Please check the [matformer-expample-commands](./matformer-example-commands) for pretraining baseline and MatFormer models along with finetuning on a released checkpoint.

## MatFormer-OLMo Checkpoints

| Name | #Parameters | #Tokens | Checkpoint |
|------| :-: | :-: | :-: |
| MatFormer-OLMo-180M | 180M | 20B | [Link](https://drive.google.com/drive/folders/1hI8wlHzQYRLfC4XdnS5Xl1vwV8S2UA0f?usp=sharing) |
| MatFormer-OLMo-460M | 460M | 40B | [Link](https://drive.google.com/drive/folders/1qi2cu9BRsE5wmZ10pdZHqffIGdo4NQRm?usp=sharing) |
| MatFormer-OLMo-1300M | 1.3B | 160B | [Link](https://drive.google.com/drive/folders/1CrgGX7iUB3tipcngc80WcG7Hf33qEzEw?usp=sharing) |
----

You can load a checkpoint like this:

```python
from olmo import Olmo, Tokenizer

checkpoint = "MatFormer-OLMo-1300M"
model = Olmo.from_checkpoint(checkpoint, device="cuda")
tokenizer = Tokenizer.from_checkpoint(checkpoint)
```

## Generating text

You can use the `generate()` method to produce text using beam search with a variety of options.

For example:

```python
# Prepare inputs.
# Note: we don't want the EOS token added to the end of the input, hence
# the `add_special_tokens=False`.
input_ids = tokenizer.encode("I'm a large language model, ", add_special_tokens=False)
# `model.generate()` expects a batch.
input_tensor = torch.tensor(input_ids).unsqueeze(0)

# Run beam search.
outputs = model.generate(input_tensor, max_steps=3, beam_size=3)

# The output token IDs are shape (batch_size, beam_size, max_steps)
best_generation = outputs.token_ids[0][0].tolist()
print(tokenizer.decode(best_generation))
```
