import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.datasets import IMDB
from torch.utils.data.dataset import random_split
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import spacy
nlp = spacy.load('en_core_web_sm')

SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
random_state = random.seed(SEED)

def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x


train_iter, test_iter = IMDB(
    split=('train', 'test'))

train_dataset = list(train_iter)
test_data = list(test_iter)

num_train = int(len(train_dataset) * 0.70)
train_data, valid_data = \
    random_split(train_dataset, 
        [num_train, 
         len(train_dataset) - num_train])

tokenizer = get_tokenizer('spacy') 
counter = Counter()
for (label, line) in train_data:
    counter.update(generate_bigrams(
        tokenizer(line))) 
vocab = Vocab(counter, 
              max_size = 25000, 
              vectors = "glove.6B.100d", 
              unk_init = torch.Tensor.normal_)

text_pipeline = lambda x: [vocab[token] 
      for token in generate_bigrams(tokenizer(x))]

label_pipeline = lambda x: 1 if x=='pos' else 0

device = torch.device("cuda" if 
    torch.cuda.is_available() else "cpu")

def collate_batch(batch):
   label_list, text_list = [], []
   for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text))
        text_list.append(processed_text)
   return (torch.tensor(label_list, dtype=torch.float64).to(device), 
          pad_sequence(text_list, 
                       padding_value=1.0).to(device))

batch_size = 64    
def batch_sampler():
    indices = [(i, len(tokenizer(s[1]))) for i, s in enumerate(train_dataset)]
    random.shuffle(indices)
    pooled_indices = []
    # create pool of indices with similar lengths 
    for i in range(0, len(indices), batch_size * 100):
        pooled_indices.extend(sorted(indices[i:i + batch_size * 100], key=lambda x: x[1]))

    pooled_indices = [x[0] for x in pooled_indices]

    # yield indices for current batch
    for i in range(0, len(pooled_indices), batch_size):
        yield pooled_indices[i:i + batch_size]

BATCH_SIZE = 64

train_dataloader = DataLoader(train_data,
                  # batch_sampler=batch_sampler(),
                  collate_fn=collate_batch,
                  batch_size=BATCH_SIZE,
                  shuffle=True)
                  # collate_fn=collate_batch)
valid_dataloader = DataLoader(valid_data, 
                  batch_size=BATCH_SIZE,
                  shuffle=True, 
                  collate_fn=collate_batch)
test_dataloader = DataLoader(test_data, 
                  batch_size=BATCH_SIZE,
                  shuffle=True, 
                  collate_fn=collate_batch)

class FastText(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 embedding_dim, 
                 output_dim, 
                 pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=pad_idx)
        self.fc = nn.Linear(embedding_dim, 
                            output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.permute(1, 0, 2)
        pooled = F.avg_pool2d(
            embedded, 
            (embedded.shape[1], 1)).squeeze(1) 
        return self.fc(pooled)

model = FastText(
            vocab_size = len(vocab), 
            embedding_dim = 100, 
            output_dim = 1, 
            pad_idx = \
              vocab['<pad>'])

pretrained_embeddings = vocab.vectors 
model.embedding.weight.data.copy_(
                    pretrained_embeddings) 

EMBEDDING_DIM = 100
unk_idx = vocab['<unk>']
pad_idx = vocab['<pad>']
model.embedding.weight.data[unk_idx] = \
      torch.zeros(EMBEDDING_DIM)          
model.embedding.weight.data[pad_idx] = \
      torch.zeros(EMBEDDING_DIM)

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)

for epoch in range(5):
  epoch_loss = 0
  epoch_acc = 0
  
  model.train()
  for label, text in train_dataloader:
      optimizer.zero_grad()
      predictions = model(text).squeeze(1)
      loss = criterion(predictions, label)
      
      rounded_preds = torch.round(
          torch.sigmoid(predictions))
      correct = \
        (rounded_preds == label).float()
      acc = correct.sum() / len(correct)
      
      loss.backward()
      optimizer.step()
      epoch_loss += loss.item()
      epoch_acc += acc.item()

  print("Epoch %d Train: Loss: %.4f Acc: %.4f" %
          (epoch,
          epoch_loss / len(train_dataloader), 
          epoch_acc / len(train_dataloader)))

  epoch_loss = 0
  epoch_acc = 0
  model.eval()
  with torch.no_grad():
    for label, text in valid_dataloader:
      predictions = model(text).squeeze(1)
      loss = criterion(predictions, label)
      
      rounded_preds = torch.round(
          torch.sigmoid(predictions))
      correct = \
        (rounded_preds == label).float()
      acc = correct.sum() / len(correct)
      
      epoch_loss += loss.item()
      epoch_acc += acc.item()

  print("Epoch %d Valid: Loss: %.4f Acc: %.4f" %
          (epoch,
          epoch_loss / len(valid_dataloader), 
          epoch_acc / len(valid_dataloader)))

test_loss = 0
test_acc = 0
model.eval()
with torch.no_grad(): 
  for label, text in test_dataloader:
    predictions = model(text).squeeze(1)
    loss = criterion(predictions, label)
    
    rounded_preds = torch.round(
        torch.sigmoid(predictions))
    correct = \
      (rounded_preds == label).float()
    acc = correct.sum() / len(correct)

    test_loss += loss.item()
    test_acc += acc.item()

print("Test: Loss: %.4f Acc: %.4f" %
        (test_loss / len(test_dataloader), 
        test_acc / len(test_dataloader)))

def predict_sentiment(model, sentence):
    model.eval()
    text = torch.tensor(text_pipeline(sentence)).unsqueeze(1).to(device)
    prediction = torch.sigmoid(model(text))
    return prediction.item()

sentiment0 = predict_sentiment(model, 
                  "Don't waste your time")
print(sentiment0)

sentiment1 = predict_sentiment(model, 
                  "You gotta see this movie!")
print(sentiment1)
