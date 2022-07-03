# Retea LSTM
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import spacy
import random
import time
from parametrii import*
import torch.nn.functional as F

#from torch.utils.tensorboard import SummaryWriter

# Date
from tokenizers import Tokenizer 
tokenizer = Tokenizer.from_file("tokenizer.json")
with open('propozitii_e100.txt', 'r', encoding='utf-8') as f: 
    sursa = [next(f) for p in range(50000)]
with open('propozitii_50.txt', 'r', encoding='utf-8') as g:
    target = [next(g) for p in range(50000)] 


# Constructie Model S2S
class Encoder(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.emb = nn.Embedding (input_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers, bidirectional=True, dropout=dropout)
        self.fc_hidden = nn.Linear(hidden_size*2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size*2, hidden_size)
    
    def forward(self, x):
        embedding = self.dropout(self.emb(x))
        encoder_states, (hidden, cell) = self.lstm(embedding)
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))

        return encoder_states, hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, output_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.emb = nn.Embedding(input_size, emb_size)
        self.lstm = nn.LSTM(hidden_size*2+emb_size, hidden_size, num_layers, dropout=dropout)
    

        self.energy = nn.Linear(hidden_size*3, 1)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)
    
    
    def forward(self, x, encoder_states, hidden, cell):
        x = x.unsqueeze(0)
        embedding = self.dropout(self.emb(x))
        sequence_length = encoder_states.shape[0]
        h_reshape = hidden.repeat(sequence_length, 1, 1)
        energy = self.relu(self.energy(torch.cat((h_reshape, encoder_states), dim=2)))
        attention = self.softmax(energy)
        attention = attention.permute(1, 2, 0)
        encoder_states = encoder_states.permute(1, 0, 2)
        context_vector = torch.bmm(attention, encoder_states).permute(1, 0, 2)
        rnn_input = torch.cat((context_vector, embedding), dim=2)
        outputs, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell
    

class S2S(nn.Module):
    def __init__(self, encoder, decoder):
        super(S2S, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, sursa, target, teacher_force_ratio = 1): 
        batch_size = sursa.shape[1]
        target_len = target.shape[0]
        target_vocab_size = 10000 # nr de tokeni

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        encoder_states, hidden, cell = self.encoder(sursa)

        x = target[0]
        for i in range(1, target_len):
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)
            outputs[i] = output
            best_guess = output.argmax(1)

            x = target[i] if random.random()<teacher_force_ratio else best_guess
        
        return outputs

encoder = Encoder(input_size_encoder, emb_size, hidden_size, num_layers, dropout).to(device)
decoder = Decoder(input_size_decoder, emb_size, hidden_size, output_size, num_layers, dropout).to(device)
model = S2S(encoder, decoder).to(device)


optimizer = optim.Adam(params=model.parameters(), lr=lr)
pad_idx = 0
criteriu = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)

#writer = SummaryWriter("runs/loss_plot") 
#step = 0  # Pt afisare grafic
from Functii import init_weights, returnMaxLen, padding, epoch_time, showPlot, corectieL
model.apply(init_weights)
propozitii = target.copy()
prop_eron = sursa.copy()

# Pentru testare model antrenat
"""
checkpoint = torch.load("name.pth.tar")
model.load_state_dict(checkpoint['state_dict'])

model.eval()
s1 = "..." # Se introduce fraza
c1 = correct_sentence(model, s1, device, max_length = len(tokenizer.encode(s1, add_special_tokens=False).ids))
print(s1, '->',c1)
"""

# Pentru antrenare
"""

pierderi = []
for epoch in range(epochs):
    print(f'Epoch [{epoch} / {epochs}]')
    losses = []
    start = time.time()
    model.train()
    
    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    save_checkpoint(checkpoint)
    
    for i in range(len(propozitii)):
        input_data = torch.tensor([])
        target_data = torch.tensor([])
        if i % lenn == 0:
            target = propozitii[i: i + lenn]
            target= list(map(lambda s: s.strip(), target))
            sursa = prop_eron[i: i + lenn]
            max_t = returnMaxLen(target, True)
            max_s = returnMaxLen(sursa, True)
            # Fac aceaa testare pt a adauga in mod eficient padding ( strict cat e nevoie)
            if max_t>max_s:
                maxx_len = max_t
            else:
                maxx_len = max_s
            tensor_correct = padding(target, True, lenn, maxx_len)
            tensor_error = padding(sursa, True, lenn, maxx_len)
        
            input_data = torch.cat((input_data, tensor_error), 1)
            input_data = input_data.long().to(device)
          
            target_data = torch.cat((target_data, tensor_correct), 1)
            target_data = target_data.long().to(device)
            
            output = model(input_data, target_data)
   
            output = output[1:].reshape(-1, output.shape[1]) 
            target_data = target_data[0:].reshape(-1)
         
            optimizer.zero_grad()
            loss = criteriu(output, target_data)
    
            losses.append(loss.item())
            loss.backward()
           
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
           
    mean_loss = sum(losses) / len(losses)
    pierderi.append(mean_loss)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start, end_time)
    print(f'Epoch: {epoch} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'Loss.ul la epoca {epoch} este {mean_loss:.5f}')
showPlot(pierderi) 
"""