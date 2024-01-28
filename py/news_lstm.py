import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from flask import Flask
from flask_restful import Api, Resource, reqparse
from torch.utils.data import DataLoader, TensorDataset
from torchtext.data.utils import get_tokenizer, ngrams_iterator
from torchtext.datasets import text_classification

NGRAMS = 1
VOCAB_SIZE = 95812
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_length = 100

# First checking if GPU is available
train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    print("Training on GPU.")
else:
    print("No GPU available, training on CPU.")

# Define the network architecture


class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.2)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)
        # print(batch_size)

        # embeddings and lstm_out
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)

        # reshape to be batch_size first
        out = out.view(batch_size, seq_length, -1)
        # print(soft_out.shape)
        out = out[:, -1, :]  # get last batch of labels

        # return last sigmoid output and hidden state
        return out, hidden

    def init_hidden(self, batch_size):
        """Initializes hidden state"""
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if train_on_gpu:
            hidden = (
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
            )
        else:
            hidden = (
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
            )

        return hidden


# Instantiate the model w/ hyperparams
vocab_size = VOCAB_SIZE
output_size = 4
embedding_dim = 512
hidden_dim = 256
n_layers = 3

net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

net.load_state_dict(torch.load("./states/newsclf.pth", map_location=device))
net.eval()


ag_news_label = {1: "World", 2: "Sport", 3: "Business", 4: "Sci/Tech"}


def predict(text, model, vocab, ngrams):
    tokenizer = get_tokenizer("basic_english")

    with torch.no_grad():
        text_ints = np.array([vocab[token] for token in ngrams_iterator(tokenizer(text), ngrams)])
        features = np.array([1] * seq_length)
        features[-len(text_ints) :] = text_ints[:seq_length]
        features = torch.from_numpy(features).unsqueeze(0).to(device)

        # initialize hidden state
        h = net.init_hidden(1)
        output, h = model(features, h)
        confidence = nn.functional.softmax(output.squeeze(), dim=0)
        return output.argmax(1).item() + 1, confidence.max().item()


vocab = torch.load("./variables/vocab.pt")
net.to(device)

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument("text")


class HelloWorld(Resource):
    def get(self):
        return {"hello": "world"}


class NewsClf(Resource):
    def post(self):
        args = parser.parse_args()
        text = args["text"]
        result, confidence = predict(text, net, vocab, NGRAMS)
        return {"result": ag_news_label[result], "confidence": confidence}, 200


# api.add_resource(HelloWorld, '/')
# api.add_resource(NewsClf, '/predict')

# if __name__ == '__main__':
#     app.run(debug=True, port=9999)
