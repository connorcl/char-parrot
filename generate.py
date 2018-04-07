import argparse

from load_config import load_config
import model


def main():
    parser = argparse.ArgumentParser(
            description="""Parrot: a character-level LSTM-RNN-based language 
            model implemented with PyTorch [Text generation script]""")
    parser.add_argument("config_file",
                        help="""File containing the model configuration. See
                             sample_config.py for a commented example""")
    parser.add_argument("-l", "--load-file",
                        help="""Load previously saved model state from LOAD_FILE. 
                        The current configuration must be consistent with that 
                        of the model which generated this file""",
                        required=True)
    parser.add_argument("-s", '--seed',
                        help="""Seed used to predict the first character.
                        Must be at least as long as the number of time steps
                        specificed in the config file""",
                        required=True)
    parser.add_argument("-n", "--length",
                        help="Length of sequence to predict and print.",
                        required=False,
                        default=250)
    parser.add_argument("-t", "--temperature",
                        help="""Temperature to use when predicting the
                        next character. Lower is more greedy, higher is
                        more random""",
                        required=False,
                        default=1)
    
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
                              save_file=None)

    charLSTM.load(args.load_file, True)
    
    charLSTM.generate(args.seed, int(args.length), config.time_steps, float(args.temperature))
    
if __name__ == "__main__":
    main()
