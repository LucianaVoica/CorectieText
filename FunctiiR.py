import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
from os import environ
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file('tokenizer.json')

def init_weights(model): #  functie de initializare ponderi
    if isinstance(model, nn.Linear):
        if model.weight is not None:
            nn.init.kaiming_uniform_(model.weight.data)
        if model.bias is not None:
            nn.init.normal_(model.bias.data)
    elif isinstance(model, nn.BatchNorm1d):
        if model.weight is not None:
            nn.init.normal_(model.weight.data, mean=1, std=0.02)
        if model.bias is not None:
            nn.init.constant_(model.bias.data, 0)
    elif isinstance(model, nn.BatchNorm2d):
        if model.weight is not None:
            nn.init.normal_(model.weight.data, mean=1, std=0.02)
        if model.bias is not None:
            nn.init.constant_(model.bias.data, 0)
    elif isinstance(model, nn.BatchNorm3d):
        if model.weight is not None:
            nn.init.normal_(model.weight.data, mean=1, std=0.02)
        if model.bias is not None:
            nn.init.constant_(model.bias.data, 0)
    else:
        pass

def epoch_time(start_time, end_time): # functie de timp
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
    
def save_checkpoint(state, filename = "name.pth.tar"):
    torch.save(state, filename)

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()

def returnMaxLen(sentences, special_tokens):

    input_ids = torch.tensor(tokenizer.encode(sentences[0], add_special_tokens=special_tokens).ids).unsqueeze(0)
    max_len = input_ids.size()[1]

    for i in range(1, len(sentences)):
        input_ids1= torch.tensor(tokenizer.encode(sentences[i], add_special_tokens = special_tokens).ids).unsqueeze(0)

        if input_ids1.size()[1] >= max_len:
            max_len = input_ids1.size()[1]

    return max_len 

def padding(propozitii, special_tokens, length, maxx_len):
    
    sent_tensor = torch.zeros((length, maxx_len))
    for i in range(length):
        initial_sentance = torch.tensor(tokenizer.encode(propozitii[i], add_special_tokens=special_tokens).ids).unsqueeze(0)
        sentance_padding = torch.stack([torch.cat([i, i.new_zeros(maxx_len - i.size(0))], 0) for i in initial_sentance],1)
        sent_tensor[i] = torch.reshape(sentance_padding, (0, sentance_padding.size()[0]))

    return torch.transpose(sent_tensor,0,1)


def corectieL(model, sentence, device, max_length):
    
   
    sentence_tensor = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True).ids).unsqueeze(1).long().to(device)
    #print(sentence_tensor)
    with torch.no_grad():
        outputs_encoder, hidden, cell = model.encoder(sentence_tensor)

    outputs = [2] ## Se primeste tokenul care marcheaza inceputul frazei

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]])
        #print(previous_word)
        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, outputs_encoder, hidden, cell)
            best_guess = output.argmax(1).item()
        outputs.append(best_guess)
    
        if output.argmax(1).item() == 3: # Atunci cand intalneste tokenul [SEP], separator de propozitii
            break

    corrected_sentence = tokenizer.decode(outputs, skip_special_tokens=True)
    return corrected_sentence




