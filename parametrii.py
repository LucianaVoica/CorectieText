# Parametrii LSTM

# Hiperparametrii de antrenare
epochs = 300 # ( se modifica)
batch_size = 4
lr = 0.0001

# Hiperparaetrii de model
device ="cpu" # sau cuda
input_size_encoder = 10000
input_size_decoder = 10000
output_size = 10000

emb_size = 300
hidden_size = 512

num_layers = 1 #( nr starturi ascunse)
dropout = 0.1
lenn = 25 # 32 (mini_batch)