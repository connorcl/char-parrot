import random
import torch


class CharData:
    """Sequence of characters with methods to convert between 
    characters and one-hot-encoded and index tensors"""
    
    def __init__(self, text):
        self.text = list(text)
        self.characters = []
        for c in self.text:
            if c not in self.characters:
                self.characters.append(c)
        self.characters.sort()
        self.nb_characters = len(self.characters)
        
    def __len__(self):
        return len(self.text)
    
    def to_tensor(self, char, tensor_format='one-hot'):
        """Convert a character to a tensor in either one-hot
        or index (target) format"""
        i = self.characters.index(char)
        if tensor_format == 'one-hot':
            tensor = torch.zeros(self.nb_characters)
            tensor[i] = 1
        elif tensor_format == 'index':
            tensor = torch.LongTensor([i])
        return tensor
    
    def from_tensor(self, tensor):
        """Convert a tensor in either one-hot or index (target)
        format to a character"""
        if tensor.size()[0] == 1:
            i = tensor[0]
        elif tensor.size()[0] == self.nb_characters:
            i = tensor.max(0)[1][0]
        char = self.characters[i]
        return char
    
    def make_sequence(self, string):
        """Convert a text string into a sequence tensor"""
        sequence = [self.to_tensor(char, 'one-hot').unsqueeze(0) for char in string]
        sequence = torch.cat(sequence, 0)
        return sequence

    
class DataLoader:
    """Generic data loader class which returns the next value from
    its data iterator when called"""

    def __init__(self, data):
        self.data = data
        self.data_iter = iter(self.data)

    def __len__(self):
        return len(self.data)
    
    def __call__(self):
        """Return the next data point from the object's iterator"""
        return next(self.data_iter)

    def reset(self):
        """Reset the object's iterator"""
        self.data_iter = iter(self.data)

        
class CharDataLoader(DataLoader):
    """Data loader to generate batches of sequential character data
    from a CharData object"""
    
    def __init__(self, data, time_steps, batch_size):
        self.data = data
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.data_points = len(self.data) - self.time_steps
        self.batches = self.data_points // self.batch_size
        self.batch_mismatch = self.data_points % self.batch_size
        if self.batch_mismatch != 0:
            self.batches += 1
        self.next_batch = 1
        self.data_iter = iter(enumerate(self.data.text[self.time_steps:]))
        
    def __len__(self):
        return self.batches
    
    def _list_to_batch(self, sequences, new_dim=True):
        """Create a batch tensor from a list of tensors, optionally
        along a new batch dimension"""
        if new_dim:
            sequences = [s.unsqueeze(0) for s in sequences]
        batch = torch.cat(sequences, 0)
        return batch
    
    def _sequential_selection(self):
        """Return the next data point and its index from the object's
        data iterator"""
        return next(self.data_iter)

    def _random_selection(self):
        """Return a random data point and its index from the same
        range as the object's data iterator"""
        selection_text = self.data.text[self.time_steps:]
        i = random.randint(0, len(selection_text)-1)
        c = selection_text[i]
        return i, c
    
    def _make_batch(self, size, selector_fn):
        """Return a batch of sequences and a corresponding batch of targets"""
        batch_sequences = []
        batch_targets = []
        sequences_append = batch_sequences.append
        targets_append = batch_targets.append
        for _ in range(size):
            i, c = selector_fn()
            sequence = self.data.make_sequence(self.data.text[i:i+self.time_steps])
            target = self.data.to_tensor(c, 'index')
            sequences_append(sequence)
            targets_append(target)
        batch_sequences = self._list_to_batch(batch_sequences, new_dim=True)
        batch_targets = self._list_to_batch(batch_targets, new_dim=False)
        return batch_sequences, batch_targets
    
    def _make_fill_batch(self):
        """Create a batch of random data points of a size suitable 
        for filling out any batch size mismatch in the final batch"""
        fill_size = self.batch_size - self.batch_mismatch
        batch_sequences, batch_targets = self._make_batch(fill_size, self._random_selection)
        return batch_sequences, batch_targets
    
    def __call__(self):
        """Return a batch of sequences and a corresponding batch of targets"""
        bs = self.batch_size if self.next_batch < self.batches or self.batch_mismatch == 0 else self.batch_mismatch
        batch_sequences, batch_targets = self._make_batch(bs, self._sequential_selection)
        
        if bs == self.batch_mismatch:
            fill_batch, fill_targets = self._make_fill_batch()
            batch_sequences = self._list_to_batch([batch_sequences, fill_batch], new_dim=False)
            batch_targets = self._list_to_batch([batch_targets, fill_targets], new_dim=False)
            
        if self.next_batch == self.batches:
            self.next_batch = 0
        else:
            self.next_batch += 1
        
        return batch_sequences, batch_targets
            
    def reset(self):
        """Reset the object's batch counter and iterator"""
        self.next_batch = 1
        self.data_iter = iter(enumerate(self.data.text[self.time_steps:]))
