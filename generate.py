import os
import argparse

from load_config import load_config
import model


def main():
    parser = argparse.ArgumentParser(
            description="""char-parrot: a character-level language model 
                        using a GRU- or LSTM-based RNN, implemented with PyTorch 
                        [Text generation script]""")
    parser.add_argument("project_dir",
                        help="""Path to the project directory containing the
                             relevant model.ini configuration file. See 
                             sample_project/model.ini for a commented example""")
    parser.add_argument("-l", "--load-file",
                        help="""Load previously saved model state from 
                             project_dir/LOAD_FILE. The current configuration 
                             must be consistent with that of the model 
                             which generated this file""",
                        required=True)
    parser.add_argument("-s", '--seed',
                        help="""Seed used to predict the first character.
                             Must be at least as long as the number of time steps
                             specified in the config file""",
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
    
    os.chdir(args.project_dir)
    config = load_config()

    char_parrot = model.CharParrot(model_type=config['model_type'],
                                   dataset_file=config['dataset_file'],
                                   case_sensitive=bool(int(config['case_sensitive'])),
                                   time_steps=int(config['time_steps']),
                                   batch_size=int(config['batch_size']),
                                   hidden_size=int(config['hidden_size']),
                                   nb_layers=int(config['nb_layers']),
                                   dropout=float(config['dropout']),
                                   learning_rate=float(config['learning_rate']),
                                   zero_hidden=bool(int(config['zero_hidden'])),
                                   save_file=None)

    char_parrot.load(args.load_file, True)
    
    char_parrot.generate(args.seed, int(args.length), int(config['time_steps']), float(args.temperature))
    
if __name__ == "__main__":
    main()
