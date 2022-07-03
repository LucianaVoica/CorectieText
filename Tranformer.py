import math
from typing import Tuple
import torch
import spacy
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

from tokenizers import Tokenizer

from torch.utils.tensorboard import SummaryWriter




class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


#from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Citirea datelor de antrenament
with open('train.txt', 'r', encoding='utf-8') as ff:
    train_iter = [next(ff) for p in range(8000)]  ## Propozitii corecte - propozitii eronate


import time
tokenizer = spacy.load("ro_core_news_sm")
#tokenizer = BertTokenizer.from_pretrained('dumitrescustefan/bert-base-romanian-cased-v1')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter)) 
#tokenizer = Tokenizer.from_file("tokenizer.json")
device = 'cuda'
"""
def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    # Convertire text in Tensor
    #data = [torch.tensor(tokenizer.encode(item), dtype=torch.long) for item in raw_text_iter]
    data = [torch.tensor(tokenizer.encode(item, add_special_tokens=True).ids).unsqueeze(1).long().to(device) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
"""
def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    # Convertire text in Tensor
    data = [torch.tensor(vocab.lookup_indices(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

# Se face recitirea datelor de antrenament pt ca au fost consumate la construirea vocabularului
with open('train.txt', 'r', encoding='utf-8') as ff:
    train_iter = [next(ff) for p in range(100000)] 


with open('validare.txt', 'r', encoding='utf-8') as ff:
    val_iter = [next(ff) for p in range(10000)] 

train_data = data_process(train_iter)
val_data = data_process(val_iter)


def batchify(data: Tensor, bsz: int) -> Tensor:
    """ Se impart datele in secvente separate bsz, eliminand elemtele suplimentare care nu 
    s-ar potrivi perfect
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_data, batch_size)  
val_data = batchify(val_data, eval_batch_size)

bptt = 35
def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:


    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

ntokens =len(vocab) 
emsize = 200 
d_hid = 200  
nlayers = 2  
nhead = 2  
dropout = 0.2 
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

import copy
import time

criterion = nn.CrossEntropyLoss()
lr = 0.0001  # learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

def train(model: nn.Module) -> None:
    model.train()  
    total_loss = 0.
    log_interval =  100 #200
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(bptt).to(device)

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        batch_size = data.size(0)
        if batch_size != bptt:
            src_mask = src_mask[:batch_size, :batch_size]
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

            


# Evaluare 
def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    model.eval()  
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0), bptt):
            data, targets = get_batch(eval_data, i)
            batch_size = data.size(0)
            if batch_size != bptt:
                src_mask = src_mask[:batch_size, :batch_size]
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += batch_size * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)


best_val_loss = float('inf')
epochs = 500
best_model = None

writer = SummaryWriter("runs/loss_plotT3")
step = 0

def save_checkpoint(state, filename = "nameT.pth.tar"):
    torch.save(state, filename)

for epoch in range(1, epochs + 1):
    print(f'Epoch [{epoch} / {epochs}]')
    epoch_start_time = time.time()
    train(model)
  
    # Salvare model
    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    save_checkpoint(checkpoint)

    val_loss = evaluate(model, val_data)  #se face evaluare in timpul antrenamentului
    valid_p.append(val_loss)
    val_ppl = math.exp(val_loss)
    elapsed = time.time() - epoch_start_time
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'valid loss {val_loss:5.2f}')
    print('-' * 89)


    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
    

    writer.add_scalar('Valid loss', val_loss, global_step=step)
    step += 1
   # scheduler.step()

