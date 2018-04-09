import argparse

from load_config import load_config
import model


def main():
    parser = argparse.ArgumentParser(
            description="""char-parrot: a character-level language model 
                        using a GRU- or LSTM-based RNN, implemented with PyTorch 
                        [Training script]""")
    parser.add_argument("config_file",
                        help="""Path to the python file containing the model
                             configuration. The .py suffix is optional, and 
                             since this path will be loaded as a module, it 
                             must be within the current directory. See 
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

    char_parrot = model.CharParrot(model_type=config.model_type,
                                   dataset_file=config.dataset_file,
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
        char_parrot.load(args.load_file)

    char_parrot.train(int(args.epochs))

if __name__ == "__main__":
    main()
