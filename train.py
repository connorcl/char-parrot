import argparse

import model
from load_config import load_config


def main():
    parser = argparse.ArgumentParser(
            description="""Parrot: a character-level LSTM-RNN-based language 
                        model implemented with PyTorch [Training script]""")
    parser.add_argument("config_file",
                        help="""File containing the model configuration. See
			     sample_config.py for a commented example""")
    parser.add_argument("-e", "--epochs",
                        help="Number of training epochs",
                        required=False,
                        default=10)
    parser.add_argument("-s", "--save-file",
                        help="""Save model state to SAVE_FILE after every epoch, 
                             overwriting any existing file""",
                        required=False)
    parser.add_argument("-l", "--load-file",
                        help="""Load model state from LOAD_FILE. The 
                             current configuration must be consistent with 
                             that of the model which generated this file""",
                        required=False)

    args = parser.parse_args()
    
    config = load_config(args.config_file)
    
    charLSTM = model.CharLSTM(dataset_file=config.dataset_file,
                              case_sensitive=config.case_sensitive,
                              time_steps=config.time_steps,
                              batch_size=config.batch_size,
                              hidden_size=config.hidden_size,
                              nb_layers=config.nb_layers,
                              dropout=config.dropout,
                              learning_rate=config.learning_rate,
                              zero_hidden=config.zero_hidden,
                              save_file=args.save_file)
    if args.load_file:
        charLSTM.load(args.load_file)
    
    charLSTM.train(int(args.epochs))
    
if __name__ == "__main__":
    main()
