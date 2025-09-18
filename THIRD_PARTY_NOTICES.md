# Third Party Notices

This project uses the following open source components:

## Python Libraries

### PyTorch

* **License** : BSD-style
* **Copyright** : Copyright (c) 2016-present, Facebook Inc.
* **Repository** : https://github.com/pytorch/pytorch

### Hugging Face Transformers

* **License** : Apache License 2.0
* **Copyright** : Copyright 2018- The Hugging Face team.
* **Repository** : https://github.com/huggingface/transformers

### Sentence Transformers

* **License** : Apache License 2.0
* **Copyright** : Copyright (c) 2019-present, Nils Reimers
* **Repository** : https://github.com/UKPLab/sentence-transformers

### FAISS

* **License** : MIT License
* **Copyright** : Copyright (c) Facebook, Inc. and its affiliates.
* **Repository** : https://github.com/facebookresearch/faiss

### Gradio

* **License** : Apache License 2.0
* **Copyright** : Copyright (c) Gradio Contributors
* **Repository** : https://github.com/gradio-app/gradio

### PyPDF2

* **License** : BSD License
* **Copyright** : Copyright (c) 2006-2011, Mathieu Fenniak
* **Repository** : https://github.com/py-pdf/PyPDF2

### BeautifulSoup4

* **License** : MIT License
* **Copyright** : Copyright (c) 2004-2024 Leonard Richardson
* **Repository** : https://www.crummy.com/software/BeautifulSoup/

### rank-bm25

* **License** : Apache License 2.0
* **Copyright** : Copyright (c) 2020 Dorian Brown
* **Repository** : https://github.com/dorianbrown/rank_bm25

### NumPy

* **License** : BSD License
* **Copyright** : Copyright (c) 2005-2024, NumPy Developers
* **Repository** : https://github.com/numpy/numpy

## Pre-trained Models

### all-MiniLM-L6-v2 (Default Embedding Model)

* **License** : Apache License 2.0
* **Provider** : sentence-transformers
* **Model Card** : https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
* **Note** : This model is downloaded automatically when first used

### Flan-T5 Base (Default Generation Model)

* **License** : Apache License 2.0
* **Provider** : Google
* **Model Card** : https://huggingface.co/google/flan-t5-base
* **Note** : This model is downloaded automatically when first used

## Docker Base Images

### Python 3.10 Slim

* **License** : Python Software Foundation License
* **Repository** : https://github.com/docker-library/python

## Additional Components

### OpenAI API (Optional)

* **License** : OpenAI Terms of Use
* **Documentation** : https://platform.openai.com/docs
* **Note** : Requires API key and is subject to usage fees

## License Compliance

All dependencies have been selected for compatibility with the MIT license of this project. Users should review individual licenses when deploying in production environments.

## Model Download Notice

Pre-trained models are downloaded automatically on first use. Models larger than 1GB require explicit user consent. To pre-download models, run:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

## Attribution

This project builds upon the excellent work of the open source community. We are grateful to all contributors of the libraries and models used in this system.
