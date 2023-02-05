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
from transformers import BertTokenizer, BertModel, EncoderDecoderModel
import gc

if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()
    time_start = time.time()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, output_hidden_states = True).to(device)
    llm_model = EncoderDecoderModel.from_encoder_decoder_pretrained(PRE_TRAINED_MODEL_NAME, PRE_TRAINED_MODEL_NAME,  output_hidden_states= True).to(device)
    llm_model.config.decoder_start_token_id = tokenizer.cls_token_id
    llm_model.config.pad_token_id = tokenizer.pad_token_id
    llm_model.config.is_encoder_decoder=True

    # --- Load data
    if csv == 'data_600.csv':
        data = pd.read_csv(csv, header = None).values
        data = data[1:]
        translations = pd.read_csv('translations_600.csv', header = None).values
        translations = translations[1:]
        data = data[0:len(translations)]
        data = np.concatenate((translations, data), axis = 1)
        X = data[:, 3:]
        y = data[:, :2]
    if csv == 'data_864.csv':
        data = pd.read_csv(csv, header = None).values
        data = data[1:]
        translations = pd.read_csv('translations_864.csv', header = None).values
        translations = translations[1:]
        data = data[0:len(translations)]
        data = np.concatenate((translations, data), axis = 1)
        X = data[:, 3:]
        y = data[:, :2]
    
    
    X, y = preprocess(X, y)

    # --- Combine and split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2, shuffle = True)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state = 2, shuffle = True)

    # --- Get embeddings

    train_embeddings, train_input_ids, train_words = get_embeddings(y_train[:, 0], bert_model, tokenizer, device)
    val_embeddings, val_input_ids, val_words  = get_embeddings(y_val[:, 0], bert_model, tokenizer, device)
    test_embeddings, test_input_ids, test_words  = get_embeddings(y_test[:, 0],bert_model ,tokenizer, device)

    # -- Dataset
    train_dataset = ECGDataset(
        report_labels = train_embeddings,
        disease_labels = y_train[:, 1].astype('int32'),
        signals = X_train,
        input_ids = train_input_ids,
        words = train_words
    )
    val_dataset = ECGDataset(
        report_labels = val_embeddings,
        disease_labels = y_val[:, 1].astype('int32'),
        signals = X_val,
        input_ids = val_input_ids,
        words = val_words
    )

    test_dataset = ECGDataset(
        report_labels = test_embeddings,
        disease_labels = y_test[:, 1].astype('int32'),
        signals = X_test,
        input_ids = test_input_ids,
        words = test_words
    )

    # --- Loader
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            num_workers=0,
                            shuffle = True)

    valid_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            num_workers=0,
                            shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            num_workers=0,
                            shuffle=True)

    optimizer = ScheduledOptim(
    Adam(filter(lambda x: x.requires_grad, llm_model.parameters()),
        betas=(0.9, 0.98), eps=1e-4, lr = 1e-5, weight_decay=1e-2), d_model, warm_steps)

    train_losses = []
    val_losses = []
    all_epochs = []

    for epoch in range(epochs):
        print('[ Epoch', epoch, ']')
        start = time.time()
        
        train_loss = finetune_train(train_loader, device, llm_model, optimizer, train_dataset.__len__())
        train_losses.append(train_loss)

        valid_loss = finetune_validate(valid_loader, device, llm_model, val_dataset.__len__(), tokenizer)
        val_losses.append(valid_loss)

        print('  - (Training)  loss: {loss: 8.5f},'
                  'elapse: {elapse:3.3f} min'.format(loss=train_loss,
                                                      elapse=(time.time() - start) / 60))
        print('  - (Validation)  loss: {loss: 8.5f},'
                  'elapse: {elapse:3.3f} min'.format(loss=valid_loss,
                                                      elapse=(time.time() - start) / 60))
        
        all_epochs.append(epoch)

    dic = {}

    dic['train_loss'] = train_losses
    dic['valid_loss'] = val_losses
    dic['epoch'] = all_epochs
    new_df = pd.DataFrame(dic)

    print('ALL DONE')               
    time_consume = (time.time() - time_start)
    print('total ' + str(time_consume) + 'seconds')
    fig1 = plt.figure('Figure 1')
    plt.plot(train_losses, label = 'train')
    plt.plot(val_losses, label= 'valid')
    plt.xlabel('epoch')
    plt.ylim([0.0, 2])
    plt.ylabel('loss')
    plt.legend(loc ="upper right")
    plt.title('loss change curve')

    plt.savefig('finetune_lr_curve_864.png')
    llm_model.save_pretrained("finetune_llm_model_864")

    llm_model = EncoderDecoderModel.from_pretrained("finetune_llm_model_864").to(device)
    finetune_test(test_loader, device, llm_model, test_dataset.__len__(), tokenizer)
