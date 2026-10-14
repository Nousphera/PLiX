# PLiX

**Plug-and-Play Multilingual Few-shot Spoken Words Recognition**

Aaqib Saeed · Vasileios Tsouvalas  
arXiv:2305.03058

Adding new voice commands usually requires new labeled data and task-specific training. PLiX learns a multilingual audio embedding from millions of one-second clips across 20 languages, then recognizes unseen spoken words by comparing them with prototypes formed from a few support examples, without retraining the model for each new vocabulary.

## Project resources

[Project page](https://nousphera.github.io/PLiX/) · [Research overview](https://nousphera.github.io/projects/plix/) · [A0 poster](https://nousphera.github.io/PLiX/poster.pdf) · [Paper](https://arxiv.org/abs/2305.03058)

The authors’ **pretrained models and inference instructions are released on [Hugging Face](https://huggingface.co/aaqibsaeed/plixkws)**. This repository hosts the project page, poster and citation; it does not duplicate model weights.

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

Publication and linked implementation licenses remain with their respective authors and publishers. No license for unreleased code is implied by this resource repository.
