import pandas as pd
import numpy as np
from dataset_new import *
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler, DataLoader
from config import *
from model_new import *
import os
from utils import *
from optim_new import *
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import matplotlib.pyplot as plt
import warnings
from nltk.translate.bleu_score import SmoothingFunction
from transformers import BertTokenizer, BertModel, EncoderDecoderModel, AutoTokenizer, AutoModel, RobertaModel, BartModel, BartTokenizer, RobertaTokenizer
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import gc


gc.collect()
torch.cuda.empty_cache()
time_start = time.time()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sm = SmoothingFunction().method2
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# --- Load data
if csv == 'data_600.csv':
    data = pd.read_csv(csv, header = None).values
    print('Loading ECG 600 Features')
    data = data[1:]
    translations = pd.read_csv('translations_600.csv', header = None).values
    translations = translations[1:]
    data = data[0:len(translations)]
    data = np.concatenate((translations, data), axis = 1)
    X = data[:, 3:]
    y = data[:, :2]
if csv == 'data_864.csv':
    data = pd.read_csv(csv, header = None).values
    print('Loading ECG 864 Features')
    data = data[1:]
    translations = pd.read_csv('translations_864.csv', header = None).values
    translations = translations[1:]
    data = data[0:len(translations)]
    data = np.concatenate((translations, data), axis = 1)
    X = data[:, 3:]
    y = data[:, :2]

#X = X[:2000]
#y = y[:2000]
X, y = preprocess(X,y)

#BartTokenizer.from_pretrained if Bart
#RobertaTokenizer.from_pretrained if Roberta
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

#BartModel.from_pretrained if Bart
#RobertaModel.from_pretrained if Roberta
bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, output_hidden_states = True).to(device)
llm_model = EncoderDecoderModel.from_pretrained(FINE_TUNED_LLM_MODEL,  output_hidden_states= True).to(device)
llm_model.config.decoder_start_token_id = tokenizer.cls_token_id
llm_model.config.pad_token_id = tokenizer.pad_token_id

# --- Combine and split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2, shuffle = True)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state = 2, shuffle = True)

# --- Get embeddings

test_embeddings, test_input_ids, test_words  = get_embeddings(y_test[:, 0],bert_model ,tokenizer, device)


test_dataset = ECGDataset(
    report_labels = test_embeddings,
    disease_labels = y_test[:, 1].astype('int32'),
    signals = X_test,
    input_ids = test_input_ids,
    words = test_words
)

# --- Loader
test_loader = DataLoader(dataset=test_dataset,
                        batch_size=batch_size,
                        num_workers=0,
                        shuffle=True)

if model_name == 'Transformer' or model_name == 'trans':
    model = Transformer(device=device, d_feature=SIG_LEN, d_model=d_model, d_inner=d_inner, n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)
elif model_name == 'MLP':
    model = MLP(vocab_size = SIG_LEN)
elif model_name == 'LSTM':
    model = BiLSTM(vocab_size = SIG_LEN, device = device)
elif model_name == 'resnet':
    model = ResNet1D(in_channels=1, base_filters=768, kernel_size=1, stride=2, groups = 16, n_block = 3, n_classes=class_num)
    
model = nn.DataParallel(model)
model = model.to(device)

optimizer = ScheduledOptim(
Adam(filter(lambda x: x.requires_grad, model.parameters()),
    betas=(0.9, 0.98), eps=1e-4, lr = 1e-5, weight_decay=1e-2), d_model, warm_steps)


test_model_name = f'{model_name}_model_{SIG_LEN}_1.chkpt'

chkpoint = torch.load(test_model_name, map_location='cuda')
model.load_state_dict(chkpoint['model'])
test(test_loader, device, model, llm_model, test_dataset.__len__(), tokenizer, sm)