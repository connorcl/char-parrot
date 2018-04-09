# char-parrot
A character-level language model using a GRU- or LSTM-based RNN, implemented with PyTorch  

## Installation
No installation is required to use char-parrot itself. However, some dependencies must be installed.  

### Python 3
Head to the [Python official website](https://python.org) to download and install Python 3.

### PyTorch
Once Python is installed, head to the [PyTorch official website](http://pytorch.org) to download and install PyTorch. The current char-parrot code
was tested on version 0.3.1.

### Unidecode and tqdm
Install Unidecode and tqdm using pip:
```bash
pip install unidecode tqdm
```

## Usage

Run ```python train.py config_file [options]``` to train a model, and ```python generate.py config_file [options]``` to generate text based on a previously trained model. Run each script with the ```--help``` flag for detailed information on its usage. Additionally, see ```sample_config.py``` for a sample configuration file with comments explaining each of its options. The model will run on the GPU if available, unless ```force_cpu``` is set to ```True``` in ```hw.py```.

### Examples

Train a model based on a configuration stored in ```config.py``` for 20 epochs, saving the model to ```save.pth``` after every epoch:
```
python train.py config -e 20 -s save.pth
```
Load the saved state ```save.pth``` and train for a further 10 epochs, saving the state to ```save.pth``` after every epoch:
```
python train.py config -e 10 -l save.pth -s save.pth
```
Generate 500 characters of text using the model whose state is saved in ```save.pth``` and whose configuration is stored in ```config.py```, using the seed phrase "once upon a time" and the sampling temperature 0.3:
```
python generate.py config -l save.pth -n 500 -s "once upon a time" -t 0.3
```
