import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

class Dictionary(object):
    ''' Represents our Vocabulary.'''
    def __init__(self):
        self.word2idx = {} #Map of words to indices, convert word into its index in vocabulary
        self.idx2word = [] #List of all words, can look up word by index

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        #Converting ASCII for characters we want (limited to those in english language)
        self.whitelist = [chr(i) for i in range(32, 127)]
        
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))

    def tokenize(self, path):
        '''Takes the text file and breaks it up into its individual word.'''
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r',  encoding="utf8") as f:
            tokens = 0
            for line in f:
                line = ''.join([c for c in line if c in self.whitelist])
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r',  encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                line = ''.join([c for c in line if c in self.whitelist])
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

class RNNModel(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        
        super(RNNModel, self).__init__()
        #Embedding layer going from our vocabulary size to defined embedding sizr
        self.encoder = nn.Embedding(vocab_size, embed_size) 
        #Add dropout layer to prevent overfitting - randomly zero some neurons
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, dropout=dropout)
        self.decoder = nn.Linear(hidden_size, vocab_size)

        self.init_weights()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def init_weights(self):
        initrange = 0.1
        #Sample from Uniform distribution
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, input, hidden):
        emb = self.drop1(self.encoder(input))
        #Pass in current batch and hidden input from previous timestep
        output, hidden = self.rnn(emb, hidden)
        output = self.drop2(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        # Reshape output to be time steps x batch size x vocabulary size
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.num_layers, batch_size, self.hidden_size).zero_())

def batchify(data, batch_size):
    # Find out how many batches we have from our corpus
    nbatch = data.size(0) // batch_size
    # Return narrow version of tensor that has integer mutiple of batch_siz
    data = data.narrow(0, 0, nbatch * batch_size)
    # Reshape and transpose the data so that it is nbatch x batch_size
    data = data.view(batch_size, -1).t().contiguous()
    if torch.cuda.is_available():
        data = data.cuda()
    return data

if __name__ == "__main__":

    corpus = Corpus('./data/shakespeare')
    vocab_size = len(corpus.dictionary)
    bs_train = 20       # batch size for training set
    bs_valid = 10       # batch size for validation set
    bptt_size = 35      # number of times to unroll the graph for back propagation through time
    clip = 0.25         # gradient clipping to check exploding gradient

    embed_size = 200    # size of the embedding vector
    hidden_size = 200   # size of the hidden state in the RNN 
    num_layers = 2      # number of RNN layers to use
    dropout_pct = 0.5   # %age of neurons to drop out for regularization

    train_data = batchify(corpus.train, bs_train)
    val_data = batchify(corpus.valid, bs_valid)
    model = RNNModel(vocab_size, embed_size, hidden_size, num_layers, dropout_pct)
    criterion = nn.CrossEntropyLoss()
    def get_batch(source, i):
        seq_len = min(bptt_size, len(source) - 1 - i)
        data = Variable(source[i:i+seq_len])
        target = Variable(source[i+1:i+1+seq_len].view(-1))
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        return data, target
    data, target = get_batch(train_data, 1)

    def train(data_source, lr):
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0
        #Copy of all the model parameters but initialized to 0
        hidden = model.init_hidden(bs_train)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        for batch, i in enumerate(range(0, data_source.size(0) - 1, bptt_size)):
            
            data, targets = get_batch(data_source, i)

            # This basically "detaches" the hidden state from the one from the previous iteration by creating a new Variable
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = Variable(hidden.data)
            
            if torch.cuda.is_available():
                hidden = hidden.cuda()
            
            # model.zero_grad()
            optimizer.zero_grad()
            
            #Feed in data generated from batch along with hidden state copied over from previous iteration.
            output, hidden = model(data, hidden)
            loss = criterion(output.view(-1, vocab_size), targets)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)
            
            optimizer.step()
            total_loss += len(data) * loss.data
            
        return total_loss[0] / len(data_source)

    def evaluate(data_source):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0
        hidden = model.init_hidden(bs_valid)
        
        for i in range(0, data_source.size(0) - 1, bptt_size):
            data, targets = get_batch(data_source, i, evaluation=True)
            
            if torch.cuda.is_available():
                hidden = hidden.cuda()
                
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, vocab_size)
            
            total_loss += len(data) * criterion(output_flat, targets).data
            hidden = Variable(hidden.data)
            
        return total_loss[0] / len(data_source)
    # Loop over epochs.
    best_val_loss = None
    def run(epochs, lr):
        global best_val_loss
        
        for epoch in range(0, epochs):
            train_loss = train(train_data, lr)
            val_loss = evaluate(val_data)
            print("Train Loss: ", train_loss, "Valid Loss: ", val_loss)

            if not best_val_loss or val_loss < best_val_loss:
                best_val_loss = val_loss
                print("save")
                torch.save(model.state_dict(), "./4.model.pth")
    run(1, 0.001)