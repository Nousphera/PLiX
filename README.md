# PLiX

**Plug-and-Play Multilingual Few-shot Spoken Words Recognition**

Aaqib Saeed · Vasileios Tsouvalas  
arXiv:2305.03058

Adding new voice commands usually requires new labeled data and task-specific training. PLiX learns a multilingual audio embedding from millions of one-second clips across 20 languages, then recognizes unseen spoken words by comparing them with prototypes formed from a few support examples, without retraining the model for each new vocabulary.

## Project resources

[Project page](https://nousphera.github.io/PLiX/) · [Research overview](https://nousphera.github.io/projects/plix/) · [A0 poster](https://nousphera.github.io/PLiX/poster.pdf) · [Paper](https://arxiv.org/abs/2305.03058)

The authors’ **pretrained models and inference instructions are released on [Hugging Face](https://huggingface.co/aaqibsaeed/plixkws)**. This repository hosts the project page, poster and citation; it does not duplicate model weights.

## Use the released package

The inference package is [plixkws 1.0 on PyPI](https://pypi.org/project/plixkws/).
Use a dedicated Python environment and install the PyTorch/torchaudio versions
appropriate for your platform, then:

```bash
python -m pip install plixkws==1.0
```

The package exposes `plixkws.model.load` and `plixkws.util.load_clip`.
For the multilingual base, choose `encoder_name="base", language="multi"`;
`language="en"` selects the English-only base. The small models are separate
language-specific models, not the multilingual base encoder.

Version 1.0's loader still points at Dropbox. To obtain the released weights
from Hugging Face, populate its local cache first. This pins the inspected
model-repository revision:

```bash
mkdir -p models
curl --fail --location \
  https://huggingface.co/aaqibsaeed/plixkws/resolve/80e18934275fc14ced9cde742949c00c5928be26/config.json \
  --output models/config.json
curl --fail --location \
  https://huggingface.co/aaqibsaeed/plixkws/resolve/80e18934275fc14ced9cde742949c00c5928be26/base_multi_model.pt \
  --output models/base_multi_model.pt
```

Provide your own segmented support/query WAV files. Each class must have the
same number of support clips in package 1.0, and every class must be represented:

```python
import torch
from plixkws import model, util

classes = ["open", "close"]
support = {
    "audio": torch.stack([util.load_clip("open.wav"), util.load_clip("close.wav")]),
    "labels": torch.tensor([0, 1]),
    "classes": classes,
}
query = {"audio": torch.stack([util.load_clip("query.wav")])}
recognizer = model.load(encoder_name="base", language="multi",
                        models_dir="models", device="cpu")
with torch.no_grad():
    indices = recognizer(support, query)
print([classes[i] for i in indices.tolist()])
```

`load_clip` resamples to 16 kHz, makes mono audio and pads/trims to one second.
The model embeds support and query clips, averages **support only** per class,
and predicts the nearest prototype by squared Euclidean distance. It returns
class indices, not calibrated rejection probabilities. Closed-set clip
recognition does not establish continuous-listening false-trigger performance.

The package metadata and wheel API were inspected on 2026-10-15; the published
weights were not downloaded or independently evaluated in this documentation
update. The package does not pin its PyTorch, torchaudio, timm or librosa
versions, so record your resolved environment before comparing results.
[Validation and provenance](docs/IMPLEMENTATION.md) lists the verified scope.

## Method at a glance

- **Learn across languages.** Train a prototypical network through episodes of support and query examples. Multilingual episodes teach the acoustic encoder to distinguish words across 20 training languages.
- **Build word prototypes.** Convert one-second audio clips into log-mel spectrograms and extract embeddings. Average the support embeddings for each requested word to form its prototype.
- **Recognize without retraining.** Embed an incoming query and compare it with the word prototypes using squared Euclidean distance. The nearest prototype supplies the predicted word for the current vocabulary.

## Citation

```bibtex
@misc{saeed2023plix,
  author = {Saeed, Aaqib and Tsouvalas, Vasileios},
  title = {Plug-and-Play Multilingual Few-shot Spoken Words Recognition},
  year = {2023},
  eprint = {2305.03058},
  archivePrefix = {arXiv},
  primaryClass = {eess.AS},
  url = {https://arxiv.org/abs/2305.03058}
}
```

## Materials and licensing

The upstream `plixkws` package and Hugging Face model card declare Apache-2.0. This resource repository does not relicense publications or datasets, or redistribute package code and weights. Follow each upstream artifact’s terms.
