# char-parrot
A character-level language model using a GRU- or LSTM-based RNN, implemented with PyTorch  

## Installation
No installation is required to use char-parrot itself - simply clone this repository by running `git clone https://github.com/cclaypool/char-parrot.git`. However, before using char-parrot, some dependencies must be installed.  

### Python 3
If you are using Linux, Python 3 is most likely already installed; if not, install it using your distribution's package manager. For other platforms, go to the [Python official website](https://python.org) to download and install Python 3.

### PyTorch
Once Python is installed, head to the [PyTorch official website](http://pytorch.org) for information on how to install the latest version of PyTorch.

### tqdm
[tqdm](https://pypi.org/project/tqdm/) is used to display progress bars during training. Install it using pip:
```bash
pip install tqdm
```

## Usage

From the char-parrot directory downloaded with `git clone`, run `python train.py project_dir [options]` to train a model, and `python generate.py project_dir [options]` to generate text based on a previously trained model. Run each script with the `--help` flag for detailed information on its usage.  

`project_dir` must contain a `model.ini` model configuration file: see `sample_project/model.ini` for a commented example explaining each option. 

The model will run on the GPU if available, unless `force_cpu` is set to `True` in `hw.py`.  

### Examples

Train a model based on a configuration stored in `project/model.ini` for 20 epochs, saving the model to `project/save.pth` after every epoch:
```bash
python train.py project -e 20 -s save.pth
```
Load the saved state `project/save.pth` and train for a further 10 epochs, saving the state to `project/save.pth` after every epoch:
```bash
python train.py project -e 10 -l save.pth -s save.pth
```
Generate 500 characters of text using the model whose state is saved in `project/save.pth` and whose configuration is stored in `project/model.ini`, using the seed phrase "once upon a time" and the sampling temperature 0.3:
```bash
python generate.py project -l save.pth -n 500 -s "once upon a time" -t 0.3
```
