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
from transformers import BertTokenizer, BertModel, EncoderDecoderModel, AutoTokenizer, AutoModel, BartTokenizer, BartModel
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import gc

if __name__ == '__main__':
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
        
    tokenizer = BartTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    bert_model = BartModel.from_pretrained(PRE_TRAINED_MODEL_NAME, output_hidden_states = True).to(device)
    llm_model = EncoderDecoderModel.from_pretrained(FINE_TUNED_LLM_MODEL,  output_hidden_states= True).to(device)
    llm_model.config.decoder_start_token_id = tokenizer.cls_token_id
    llm_model.config.pad_token_id = tokenizer.pad_token_id

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
    
    X = X[0:3000]
    y = y[0:3000]
    X, y = preprocess(X,y)

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

    # --- Sampler
    target = y_train[:, 1].astype('int')
    class_sample_count = np.unique(target, return_counts=True)[1]
    weight = 1. / class_sample_count
    samples_weight = weight[target]
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    # --- Loader
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            num_workers=0,
                            sampler = sampler)

    valid_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            num_workers=0,
                            shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            num_workers=0,
                            shuffle=True)

    if (model_name == 'Transformer') or (model_name == 'trans'):
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

    early_stopping = EarlyStopper(patience=5, min_delta=0.2)

    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []
    all_epochs = []

    for epoch in range(epochs):
        print('[ Epoch', epoch, ']')
        start = time.time()
        
        train_loss, train_acc, train_cm, _, _ = train(train_loader, device, model, llm_model, optimizer, train_dataset.__len__())
        train_accs.append(train_acc)
        train_losses.append(train_loss)

        valid_loss, valid_acc, valid_cm, _, _ = validate(valid_loader, device, model, llm_model, val_dataset.__len__(), tokenizer)
        
        val_accs.append(valid_acc)
        val_losses.append(valid_loss)

        model_state_dict = model.state_dict()
        
        checkpoint = {
            'model': model_state_dict,
            'config_file': 'config',
            'epoch': epoch}

        if valid_loss <= min(val_losses):
            torch.save(checkpoint, f'{model_name}_model_{SIG_LEN}_{layers}.chkpt')
            print('    - [Info] The checkpoint file has been updated.')

        print('  - (Training)  loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, '
                  'elapse: {elapse:3.3f} min'.format(loss=train_loss, accu=100 * train_acc,
                                                      elapse=(time.time() - start) / 60))
        print("train_cm:", train_cm)
        print('  - (Validation)  loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, '
                  'elapse: {elapse:3.3f} min'.format(loss=valid_loss, accu=100 * valid_acc,
                                                      elapse=(time.time() - start) / 60))
        print("valid_cm:", valid_cm)
        
        all_epochs.append(epoch)
        
        if early_stopping.early_stop(valid_loss):
            print("We are at epoch:", epoch)
            break
        

    dic = {}

    dic['train_acc'] = train_accs
    dic['train_loss'] = train_losses
    dic['valid_acc'] = val_accs
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
    plt.savefig(f'pngs/{model_name}_acc_curve_{SIG_LEN}_{layers}.png')
    
    fig2 = plt.figure('Figure 2')
    plt.plot(train_accs, label = 'train')
    plt.plot(val_accs, label= 'valid')
    plt.xlabel('epoch')
    plt.ylim([0.0, 1])
    plt.ylabel('accs')
    plt.legend(loc ="upper right")
    plt.title('accuracy change curve')

    plt.savefig(f'pngs/{model_name}_loss_curve_{SIG_LEN}_{layers}.png')

    test_model_name = f'{model_name}_model_{SIG_LEN}_{layers}.chkpt'

    chkpoint = torch.load(test_model_name, map_location='cuda')
    model.load_state_dict(chkpoint['model'])
    test(test_loader, device, model, llm_model, test_dataset.__len__(), tokenizer, sm)
    print(f'MODEL NAME: {model_name}_{SIG_LEN}_{layers}')
