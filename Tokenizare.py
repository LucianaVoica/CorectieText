

from datasets import load_dataset
dataset = load_dataset('oscar', 'unshuffled_original_ro', split='train[:50000]')


import os
os.mkdir('./oscar')

from tqdm.auto import tqdm 
text_data = []
file_count = 0


g= open('./oscar/file_{file_count}.txt', 'w', encoding='utf-8')
for sample in tqdm(dataset):
    sample = sample['text'].replace('\n',' ')
    text_data.append(sample)
    if len(text_data) == 5_000:
        g.write('\n'.join(text_data))
        text_data = []
        file_count +=1


from pathlib import Path 
paths = [ str(x) for x in Path('./oscar').glob('**/*.txt')]

from tokenizers import BertWordPieceTokenizer
from tokenizers.models import WordPiece

tokenizer = BertWordPieceTokenizer(
    clean_text = True,
    handle_chinese_chars = False, # pt ca nu lucrez cu caractere chinezesti
    strip_accents = False, # pt a lua in cosiderare si literele cu diacritice
    lowercase = True # pt a transfomma totul in litera mica
)
tokenizer.train (files =paths,
    vocab_size = 10_000,
    min_frequency = 2,
    special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]','[MASK]'],
    limit_alphabet = 1000,
    wordpieces_prefix = '##'
)

tokenizer.save("tokenizer.json")
