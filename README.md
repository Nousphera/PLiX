# PLiX

**Plug-and-Play Multilingual Few-shot Spoken Words Recognition**

Aaqib Saeed · Vasileios Tsouvalas · arXiv:2305.03058 2023

[<img src="docs/assets/icons/paper.svg" width="16" height="16" alt=""> Paper](https://arxiv.org/abs/2305.03058) · [<img src="docs/assets/icons/project.svg" width="16" height="16" alt=""> Project](https://nousphera.github.io/PLiX/) · [Poster](https://nousphera.github.io/PLiX/poster.pdf)

## Abstract

As technology advances and digital devices become prevalent, seamless human-machine communication is increasingly gaining significance. The growing adoption of mobile, wearable, and other Internet of Things (IoT) devices has changed how we interact with these smart devices, making accurate spoken words recognition a crucial component for effective interaction. However, building robust spoken words detection system that can handle novel keywords remains challenging, especially for low-resource languages with limited training data. Here, we propose PLiX, a multilingual and plug-and-play keyword spotting system that leverages few-shot learning to harness massive real-world data and enable the recognition of unseen spoken words at test-time. Our few-shot deep models are learned with millions of one-second audio clips across 20 languages, achieving state-of-the-art performance while being highly efficient. Extensive evaluations show that PLiX can generalize to novel spoken words given as few as just one support example and performs well on unseen languages out of the box. We release models and inference code to serve as a foundation for future research and voice-enabled user interface development for emerging devices.

## Install

The released inference implementation is the authors’
[plixkws package](https://pypi.org/project/plixkws/) and
[Hugging Face models](https://huggingface.co/aaqibsaeed/plixkws).
This repository provides documentation and research materials.

Use a dedicated Python 3.11 environment. The following combination was tested
on macOS arm64 CPU:

```bash
git clone https://github.com/Nousphera/PLiX.git
cd PLiX
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install plixkws==1.0 torch==2.14.1 torchvision==0.29.1 \
  torchaudio==2.11.0 timm==1.0.30 numpy==1.26.4 librosa==0.11.0
python -m pip check
```

[The complete tested environment](requirements-tested-macos-py311.txt) records
the remaining versions. On other platforms, install mutually compatible
PyTorch, torchvision and torchaudio wheels for your CPU or CUDA environment.

## Models and input audio

Download the multilingual base model from the authors’ pinned Hugging Face
revision into the package’s model cache:

```bash
mkdir -p models
curl --fail --location \
  https://huggingface.co/aaqibsaeed/plixkws/resolve/80e18934275fc14ced9cde742949c00c5928be26/config.json \
  --output models/config.json
curl --fail --location \
  https://huggingface.co/aaqibsaeed/plixkws/resolve/80e18934275fc14ced9cde742949c00c5928be26/base_multi_model.pt \
  --output models/base_multi_model.pt
python - <<'PY_CHECK'
from hashlib import sha256
from pathlib import Path
actual = sha256(Path("models/base_multi_model.pt").read_bytes()).hexdigest()
assert actual == "ed7298544e7fa5e5966ce193ba2519b12714311231c15313a7e096632fa9ac80", actual
print("Checkpoint verified")
PY_CHECK
```

Provide three separate WAV recordings in this directory: `open.wav` containing
one spoken “open”, `close.wav` containing one spoken “close”, and `query.wav`
containing a held-out example of either word. Segment each to approximately
one second with the word centered. `load_clip` averages channels, resamples to
16 kHz and pads/trims to one second. It does not find a keyword in a long recording.

For a larger vocabulary, provide the same positive number of support clips for
every class. Keep query recordings out of support. This inference example uses
your recordings and the pretrained encoder.

## Run

```bash
python - <<'PY_INFERENCE'
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
PY_INFERENCE
```

The command prints a predicted class name. The model averages support embeddings
into prototypes and chooses the closest prototype for each query. Predictions
are closed-set class indices, not calibrated rejection probabilities. Use
`language="en"` only with the separate English model; small models are also
separate language-specific artifacts.

## Configuration and outputs

Select the encoder and language when loading a model; keep class names in the
same order as the support labels. Record the package/dependency versions, model
revision, checkpoint checksum and support/query clip IDs with each experiment.
The [inference guide](docs/IMPLEMENTATION.md) details model selection, input
shapes and support-set requirements.

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

The upstream package and model card declare Apache-2.0. This documentation repository does not relicense publications or datasets or duplicate the implementation and weights. Follow the upstream artifacts’ terms.
