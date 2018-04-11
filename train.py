import os
import argparse

from load_config import load_config
import model


def main():
    parser = argparse.ArgumentParser(
            description="""char-parrot: a character-level language model 
                        using a GRU- or LSTM-based RNN, implemented with PyTorch 
                        [Training script]""")
    parser.add_argument("project_dir",
                        help="""Path to the project directory containing the
                             relevant model.ini configuration file. See 
                             sample_project/model.ini for a commented example""")
    parser.add_argument("-e", "--epochs",
                        help="Number of training epochs",
                        required=False,
                        default=10)
    parser.add_argument("-s", "--save-file",
                        help="""Save model state to project_dir/SAVE_FILE after 
                             every epoch, overwriting any existing file""",
                        required=False)
    parser.add_argument("-l", "--load-file",
                        help="""Load model state from project_dir/LOAD_FILE. The
                             current configuration must be consistent with
                             that of the model which generated this file""",
                        required=False)

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
                                   save_file=args.save_file)
    if args.load_file:
        char_parrot.load(args.load_file)

    char_parrot.train(int(args.epochs))

if __name__ == "__main__":
    main()
