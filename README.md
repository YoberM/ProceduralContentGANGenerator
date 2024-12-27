# SAGAN Implementation

This repository contains an implementation of Self-Attention Generative Adversarial Networks (SAGAN) for procedural content generation.

## Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
```

2. Activate the virtual environment:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install PyTorch:
   Visit [PyTorch's official website](https://pytorch.org/get-started/locally/) for the most up-to-date installation command. The recommended command for CUDA 12.4 is:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Usage

You can test the procedural content GAN generator by running the `SAGAN.ipynb` notebook.

## Reference

This project is based on:

Zhang, H., Goodfellow, I., Metaxas, D., & Odena, A. (2018). Self-Attention Generative Adversarial Networks. *arXiv preprint arXiv:1805.08318*.

```bibtex
@misc{zhang2018selfattention,
      title={Self-Attention Generative Adversarial Networks}, 
      author={Han Zhang and Ian Goodfellow and Dimitris Metaxas and Augustus Odena},
      year={2018},
      eprint={1805.08318},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```
