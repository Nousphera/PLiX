# Implementation entry points and verification

The release boundary is the author-maintained `plixkws` Python package and
Hugging Face model repository. This repository hosts documentation, project
materials and citation. It intentionally contains no duplicate implementation.

Verified 2026-10-15:

- [PyPI package](https://pypi.org/project/plixkws/): version 1.0, released
  2023-05-08; wheel SHA-256
  `958f42ca5975627cc955f52e39d96d58a2d2b53de83a77b7fd45640cc5d65738`.
- [Hugging Face models](https://huggingface.co/aaqibsaeed/plixkws/tree/80e18934275fc14ced9cde742949c00c5928be26):
  immutable revision `80e18934275fc14ced9cde742949c00c5928be26` contains
  `config.json`, `base_multi_model.pt`, `base_en_model.pt` and small-model files.
- The wheel's `model.py`, `util.py`, `backbone.py` and `protonet.py` were read.
  README calls match those APIs. Cache-prefetch instructions avoid dependence
  on the package's historical Dropbox download URLs.
- The historical `FewshotML/plix` GitHub homepage and `test_model.py` links from
  the upstream metadata/model card return 404. They are not valid installation
  entry points; use PyPI and Hugging Face above. This repository cannot repair
  links in another publisher's package metadata or model card.
- The package and HF model card declare Apache-2.0. Those upstream declarations
  do not change publication or dataset terms.

No model weights or user recordings were downloaded. Package installation,
full inference, recognition accuracy, quantized deployment and current-library
compatibility were not independently validated by this documentation update.
The README example was checked against the wheel API and Python syntax only.

For a reproducible inference experiment, record the package version, HF revision,
chosen encoder/language, model-file checksum, resolved dependency versions,
support/query clip IDs, class order, number of examples per class and any episode
sampling seed. Keep query clips separate from support. The supplied package
requires equal support counts because it stacks per-class embedding tensors.

The paper's keyword-disjoint training/test vocabulary and episodic benchmark
are separate from a small local inference example. Language-specific small
models and the multilingual base are separate trained artifacts. No original
training pipeline is claimed by this documentation-only repository.
