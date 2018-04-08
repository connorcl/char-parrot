import os
import sys
from unidecode import unidecode
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from chardata import CharData, CharDataLoader


class LSTMnet(nn.Module):
    """Long Short-Term Memory model"""
    
    def __init__(self, input_size, output_size, hidden_size, batch_size, nb_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.nb_layers = nb_layers
        self.hidden = (Variable(torch.zeros(self.nb_layers, self.batch_size, self.hidden_size)).cuda(),
                       Variable(torch.zeros(self.nb_layers, self.batch_size, self.hidden_size)).cuda())
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.nb_layers,
                            batch_first=True,
                            dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(self.hidden_size, output_size)
        
    def forward(self, x):
        """Forward pass for a sequence"""
        x, self.hidden = self.lstm(x, self.hidden)
        self.detach_hidden()
        x = self.dropout(x)
        x = self.out(x)
        return x
    
    def set_mode(self, mode):
        """Set the hidden size and zero the data for either batch training 
        or generation"""
        if mode == 'train':
            self.hidden = (Variable(torch.zeros(self.nb_layers, self.batch_size, self.hidden_size)).cuda(),
                           Variable(torch.zeros(self.nb_layers, self.batch_size, self.hidden_size)).cuda())
        elif mode == 'generate':
            self.hidden = (Variable(torch.zeros(self.nb_layers, 1, self.hidden_size)).cuda(),
                           Variable(torch.zeros(self.nb_layers, 1, self.hidden_size)).cuda())
    
    def detach_hidden(self, zero=False):
        """Detach the hidden state, and optionally zero the hidden data"""
        if zero:
            self.hidden = (Variable(torch.zeros(self.nb_layers, self.batch_size, self.hidden_size)).cuda(),
                           Variable(torch.zeros(self.nb_layers, self.batch_size, self.hidden_size)).cuda())
        else:
            self.hidden = tuple(Variable(v.data) for v in self.hidden)
            

class GRUnet(nn.Module):
    """Gated Recurrent Unit model"""
    
    def __init__(self, input_size, output_size, hidden_size, batch_size, nb_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.nb_layers = nb_layers
        self.hidden = Variable(torch.zeros(self.nb_layers, self.batch_size, self.hidden_size)).cuda()
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.nb_layers,
                          batch_first=True,
                          dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(self.hidden_size, output_size)
        
    def forward(self, x):
        """Forward pass for a sequence"""
        x, self.hidden = self.gru(x, self.hidden)
        self.detach_hidden()
        x = self.dropout(x)
        x = self.out(x)
        return x
    
    def set_mode(self, mode):
        """Set the hidden size and zero the data for either batch training 
        or generation"""
        if mode == 'train':
            self.hidden = Variable(torch.zeros(self.nb_layers, self.batch_size, self.hidden_size)).cuda()
        elif mode == 'generate':
            self.hidden = Variable(torch.zeros(self.nb_layers, 1, self.hidden_size)).cuda()
    
    def detach_hidden(self, zero=False):
        """Detach the hidden state, and optionally zero the hidden data"""
        if zero:
            self.hidden = Variable(torch.zeros(self.nb_layers, self.batch_size, self.hidden_size)).cuda()
        else:
            self.hidden = Variable(self.hidden.data)


class CharParrot:
    """A character-level language model using a recurrent neural network (GRU- or LSTM-based)"""
    
    def __init__(self, model, dataset_file, case_sensitive, time_steps, batch_size, hidden_size, nb_layers, dropout, learning_rate, zero_hidden, save_file):
        f = open(dataset_file, 'r')
        try:
            text = unidecode(f.read())
            if not case_sensitive:
                text = text.lower()
        finally:
            f.close()
        data = CharData(text)
        self.dataloader = CharDataLoader(data, time_steps, batch_size)
        self.save_file = save_file
        if model.lower() == "gru":
            Model = GRUnet
        elif model.lower() == "lstm":
            Model = LSTMnet
        else:
            print("No such model type!")
            exit(1)
        self.model = Model(self.dataloader.data.nb_characters, self.dataloader.data.nb_characters, hidden_size, batch_size, nb_layers, dropout).cuda()
        self.zero_hidden = zero_hidden
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate)

    def train(self, epochs=10):
        """Train the recurrent model"""
        self.model.set_mode('train')
        self.dataloader.reset()
        
        running_loss = 0.0
        
        for epoch in range(1, epochs+1):
            current_loss = running_loss/len(self.dataloader)
            if running_loss == 0.0:
                current_loss = "N/A"
            else:
                current_loss = round(current_loss, 5)
            info = "Epoch {}/{} (Current loss: {})".format(epoch, epochs, current_loss)
            running_loss = 0.0
            for i in tqdm(range(len(self.dataloader)), desc=info):
                self.model.detach_hidden(zero=self.zero_hidden)
                self.optimizer.zero_grad()
                sequence, target = self.dataloader()
                sequence, target = Variable(sequence), Variable(target)
                outputs = self.model(sequence)
                loss = self.criterion(torch.chunk(outputs, self.dataloader.time_steps, 1)[-1].squeeze(1), target)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.data[0]
            self.dataloader.reset()
            if self.save_file is not None:
                print("Saving progress...")
                self.save(self.save_file)
        print("\nDone! Final loss: %f" % (running_loss / len(self.dataloader)))
            
    def generate(self, seed, length, prev_chars, temperature=1, quiet=False):
        """Generate text using the recurrent model"""
        self.model.set_mode('generate')
        text = seed
        if not quiet: print("#" * 35 +  "\n# Generated text (including seed) #\n" + "#" * 35)
        sys.stdout.write(text)
        for _ in range(length):
            prediction_text = text[-prev_chars:]
            sequence = self.dataloader.data.make_sequence(prediction_text)
            inputs = Variable(sequence).unsqueeze(0)
            outputs = self.model(inputs)
            output = torch.chunk(outputs, prev_chars, 1)[-1].squeeze(1)
            output = output / temperature
            probs = F.softmax(output, dim=1).squeeze(0)
            prediction = probs.multinomial()
            text += self.dataloader.data.characters[prediction.data[0]]
            sys.stdout.write(text[-1])
        sys.stdout.write('\n')
        
    def save(self, save_filename, quiet=False):
        """Save the state dicts of the model and optimizer to a file"""
        torch.save({'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                }, '%s' % (str(save_filename)))
        print("Progress saved!")
    
    def load(self, load_file, quiet=False):
        """Load previously saved model and optimizer state dicts from a file"""
        if not os.path.isfile(load_file):
            print("ERROR: File does not exist")
            exit(1)
        else:
            state = torch.load(load_file)
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optimizer'])
            if not quiet: print("Model and optimizer states loaded successfully!")