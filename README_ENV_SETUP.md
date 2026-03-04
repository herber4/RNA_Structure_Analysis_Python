# Environment setup for SHAPE_Analysis_Fixed.ipynb

Recommended: use Conda (Miniconda already installed on this machine).

1) Create the environment from the YAML (recommended):

```bash
conda env create -f shape_environment.yml
conda activate shape-analysis
python -m ipykernel install --user --name shape-analysis --display-name "Python (shape-analysis)"
```

2) Alternatively, create a venv and install pip requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools whename: shape-analysis
channels:
  - cytchannels:
  - condast  - conder  - pytorch
deandependenciv   - python=3me  - pip
  - nue-  - nuis  - pand``  - matpln   - seaborn
 an  - scikithe  - scipy
  - jPy  - jupyap  - notebook
or  - ipykernha  - pip:
   ve    - n J    - torch
   
-    - torcin    - tqdm
    an     - jobrmYML

cat > ifcapipcat > ifcapipcat > ifcapipcat > ifcapipcat > ifcapipcat > ifcapipcat > ifcapipcat > ifcapipcat > ifcapipcat > ifcapipcat > ifcicon) support, adjust the `pip` torch install command as recommended by PyTorch docs.
