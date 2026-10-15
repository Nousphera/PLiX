# Inference guide

PLiX inference is provided by the authors' [plixkws package](https://pypi.org/project/plixkws/)
and [Hugging Face models](https://huggingface.co/aaqibsaeed/plixkws).
This repository contains installation instructions, research materials and citation.

## Package and models

| Resource | Version or identifier |
|---|---|
| Python package | `plixkws==1.0` |
| Model revision | `80e18934275fc14ced9cde742949c00c5928be26` |
| Multilingual base file | `base_multi_model.pt` |
| Base model SHA-256 | `ed7298544e7fa5e5966ce193ba2519b12714311231c15313a7e096632fa9ac80` |
| Base model parameters | 52,857,924 |

The [pinned model repository](https://huggingface.co/aaqibsaeed/plixkws/tree/80e18934275fc14ced9cde742949c00c5928be26)
contains `config.json`, multilingual and English base checkpoints, and separate
language-specific small models. Follow the [README download commands](../README.md#models-and-input-audio)
to populate the package cache from Hugging Face before calling `model.load`.

Use `encoder_name="base", language="multi"` for multilingual inference.
`language="en"` selects a separate English checkpoint. Small models also use
separate language-specific weights; choose a checkpoint matching both options.

## Audio and support sets

`util.load_clip(path)` averages channels, resamples to 16 kHz and pads or trims
to one second. The output shape is `(1,16000)`. Segment and center one spoken
word in each WAV before loading it. Long-recording segmentation is a separate
preprocessing step.

A support set contains audio, integer labels and ordered class names. Supply
the same positive number of examples for every class: the package stacks
per-class embedding tensors to form prototypes. Label indices must match the
class-name order. Keep query clips separate from the support set.

For two classes with one support example each, the support tensor has shape
`(2,1,16000)` and a single-query tensor has shape `(1,1,16000)`. The recognizer
returns a `(1,)` tensor of class indices. Map each index back to the same class
list used for support.

The encoder stays fixed. Class prototypes average support embeddings, and
queries are assigned to the nearest prototype by squared Euclidean distance.
Changing the support set changes the vocabulary. Predictions are closed-set
choices among supplied classes; continuous detection and unknown-word rejection
need separate application logic.

## Environment and experiment records

[`requirements-tested-macos-py311.txt`](../requirements-tested-macos-py311.txt)
records the Python 3.11/macOS arm64 CPU dependency set used for checkpoint loading
and inference. On other platforms, choose compatible PyTorch, torchvision and
torchaudio builds for the intended CPU/CUDA runtime. Package 1.0 uses PyTorch's
default checkpoint loader; the supplied base checkpoint contains tensor state
entries and works with the documented PyTorch pin.

Record package versions, model revision, encoder/language, checkpoint checksum,
class order, support/query clip IDs, examples per class and episode sampling
seed. For episodic evaluation, keep training and test vocabularies disjoint and
report the vocabulary size, support count and query count.

## Licensing

The package and Hugging Face model card declare Apache-2.0. Follow those artifact
terms and the separate terms for any publications, recordings or datasets used.
