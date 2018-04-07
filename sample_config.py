## Model configuration

# File containing the training dataset
dataset_file = "data.txt"
# Whether the training is case-sensitive
case_sensitive = True
# The number of previous characters the model will use
# to predict the next character
time_steps = 16
# The size of each batch of training examples
batch_size = 32
# The size of the hidden and cell states of the LSTM layer(s)
hidden_size = 128
# The number of LSTM layers
nb_layers = 1
# The rate of dropout after the each layer
dropout = 0.2
# The learning rate for the RMSprop optimizer
learning_rate = 0.002
# Whether to zero the hidden and cell states of the LSTM layer(s)
# after each batch of training examples
zero_hidden = True