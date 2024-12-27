# ProceduralContentGANGenerator

To Create virtual environment use
python3 -m venv venv

To run the virtual environment
source venv/bin/activate

Install all dependencies
pip install -r requirements.txt

Install torch (this is absolutely needed, there are some errors using pip)
https://pytorch.org/get-started/locally/
(Probably will you will use this command pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124)

Once you installed all dependencies you can test the procedural content gan generator on SAGAN Notebook (SAGAN.ipynb)


This project is based on:
Self-Attention Generative Adversarial Networks
Han Zhang, Ian Goodfellow, Dimitris Metaxas, Augustus Odena. 2019.
Available at: arXiv:1805.08318.
