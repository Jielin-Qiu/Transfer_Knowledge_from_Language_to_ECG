from config import *
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score
import torch.nn.functional as F
import torch
import nltk
from rouge import Rouge
from transformers import BertTokenizer, BertModel
import os
from nltk.translate.bleu_score import sentence_bleu
from scipy.stats import wasserstein_distance
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate import meteor_score
from evaluate import load
from statistics import mean
bertscore = load('bertscore')
nltk.download('wordnet')
nltk.download('omw-1.4')

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def cal_loss(pred, llm_output, report, label):

    loss1 = F.cross_entropy(pred, label, reduction='sum')
    loss2 = wasserstein_distance(llm_output.cpu().detach().numpy().flatten(), 
        report.cpu().detach().numpy().flatten())
    loss2 = torch.tensor(loss2, requires_grad=True)
    loss = loss1 + loss2
    pred = pred.max(1)[1]
    n_correct = pred.eq(label).sum().item()

    return loss, n_correct

def train(train_loader, device, model, llm_model, optimizer, total_num):
    all_labels = []
    all_res = []
    all_pred =[]
    model.train()
    total_loss = 0
    total_correct = 0    
    
    for batch in tqdm(train_loader, desc='- (Training)  '): 

        sig, report, label, input_ids = map(lambda x: x.to(device), batch[:-1])
        optimizer.zero_grad()
        if (model_name == 'LSTM') or (model_name == 'resnet'):
            sig = sig.unsqueeze(1) #for lstm and resnet
        pred, out_llm, _ = model(sig)
        if model_name == 'LSTM':
            pred = pred[:, -1, :] #for lstm
        if (model_name == 'MLP') or (model_name == 'resnet'):
            out_llm = out_llm.unsqueeze(1)#for mlp and resnet
        input_ids = input_ids[:, -1, :]
        output = llm_model(inputs_embeds = out_llm, labels = input_ids, output_hidden_states = True)
        output= output.decoder_hidden_states[0]
        all_labels.extend(label.cpu().numpy())
        all_res.extend(pred.max(1)[1].cpu().numpy())
        loss, n_correct = cal_loss(pred, output, report, label)
        all_pred.extend(pred.cpu().detach().numpy())
        loss.backward()
        optimizer.step_and_update_lr()

        total_loss += loss.item()
        total_correct += n_correct
        cm = confusion_matrix(all_labels, all_res)

    train_loss = total_loss / total_num
    train_acc = total_correct / total_num
    return train_loss, train_acc, cm, all_pred, all_labels

def validate(valid_loader, device, model, llm_model, total_num, tokenizer):
    all_labels = []
    all_res = []
    all_pred = []
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc='- (Validation)  '):
            sig, report, label, input_ids = map(lambda x: x.to(device), batch[:-1])
            if (model_name == 'LSTM') or (model_name == 'resnet'):
                sig = sig.unsqueeze(1) #for lstm and resnet
            pred, out_llm, _ = model(sig)
            if model_name == 'LSTM':
                pred = pred[:, -1, :] #for lstm
            if (model_name == 'MLP') or (model_name == 'resnet'):
                out_llm = out_llm.unsqueeze(1)#for mlp and resnet
            input_ids = input_ids[:, -1, :]
            output = llm_model(inputs_embeds = out_llm, labels = input_ids, output_hidden_states = True)
            output= output.decoder_hidden_states[0]
            
            all_labels.extend(label.cpu().numpy())
            all_res.extend(pred.max(1)[1].cpu().numpy())
            all_pred.extend(pred.cpu().numpy())
            loss, n_correct = cal_loss(pred, output, report, label)

            total_loss += loss.item()
            total_correct += n_correct

    cm = confusion_matrix(all_labels, all_res)
    acc_SP, pre_i, rec_i, F1_i = cal_statistic(cm)
    print('acc_SP is : {acc_SP}'.format(acc_SP=acc_SP))
    print('pre_i is : {pre_i}'.format(pre_i=pre_i))
    print('rec_i is : {rec_i}'.format(rec_i=rec_i))
    print('F1_i is : {F1_i}'.format(F1_i=F1_i))
    valid_loss = total_loss / total_num
    valid_acc = total_correct / total_num
    return valid_loss, valid_acc, cm, all_pred, all_labels

def test(test_loader, device, model, llm_model, total_num, tokenizer, sm):
    all_labels = []
    all_res = []
    all_pred = []
    rouge_r =[]
    rouge_p =[]
    rouge_f =[]
    list_of_hypothesis = []
    list_of_references =[]
    soft_max = []
    meteors = []
    gen_words = []
    ground_words = []
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='- (Testing)  '):
            sig, report, label, input_ids = map(lambda x: x.to(device), batch[:-1])
            words = batch[-1]
            if (model_name == 'LSTM') or (model_name == 'resnet'):
                sig = sig.unsqueeze(1) #for lstmand resnet
            pred, out_llm, soft = model(sig)
            if model_name == 'LSTM':
                pred = pred[:, -1, :] #for lstm
            if (model_name == 'MLP') or (model_name == 'resnet'):
                out_llm = out_llm.unsqueeze(1)#for mlp and resnet
            input_ids = input_ids[:, -1, :]
            output = llm_model(inputs_embeds = out_llm, labels = input_ids, output_hidden_states = True)
            output= output.decoder_hidden_states[0]
            output_ids = llm_model.generate(inputs_embeds = output)
            all_labels.extend(label.cpu().numpy())
            soft_max.extend(soft.cpu().numpy())
            all_res.extend(pred.max(1)[1].cpu().numpy())
            all_pred.extend(pred.cpu().numpy())
            r_scores = []
            p_scores = []
            f_scores = []
            generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens = True)
            for i in range(len(generated_text)):
                try:
                    print('Generated: ', generated_text[i])
                    gen_words.append(generated_text[i])
                    ground_words.append(words[i])
                    print('Ground True: ', words[i])
                    _ = sentence_calc_bleu_n(words[i].split(), generated_text[i].split(), 1)
                    list_of_hypothesis.append(generated_text[i])
                    list_of_references.append(words[i])
                    r, p, f = calc_rouge(generated_text[i], words[i])
                    meteor = meteor_score([words[i].split()], generated_text[i].split())
                    meteors.append(meteor)

                    r_scores.append(r)
                    f_scores.append(f)
                    p_scores.append(p)
                except ValueError:
                    pass
            rouge_r.append(average(r_scores))
            rouge_f.append(average(f_scores))
            rouge_p.append(average(p_scores))

            loss, n_correct = cal_loss(pred, output, report, label)

            total_loss += loss.item()
            total_correct += n_correct

    np.savetxt(f'all_pred.txt',all_pred)
    np.savetxt(f'all_label.txt', all_labels)
    all_pred = np.array(all_pred)
    cm = confusion_matrix(all_labels, all_res)
    print("test_cm:", cm)
    acc_SP, pre_i, rec_i, F1_i = cal_statistic(cm)
    print('acc_SP is : {acc_SP}'.format(acc_SP=acc_SP))
    print('pre_i is : {pre_i}'.format(pre_i=pre_i))
    print('rec_i is : {rec_i}'.format(rec_i=rec_i))
    print('F1_i is : {F1_i}'.format(F1_i=F1_i))
    test_acc = total_correct / total_num
    print('test_acc is : {test_acc}'.format(test_acc=test_acc))
    print(f'Bleu-1 Score is: {calc_bleu_n(list_of_references, list_of_hypothesis, 1)} ')
    # print(f'Bleu-2 Score is: {calc_bleu_n(list_of_references, list_of_hypothesis, 2)} ')
    # print(f'Bleu-3 Score is: {calc_bleu_n(list_of_references, list_of_hypothesis, 3)} ')
    # print(f'Bleu-4 Score is: {calc_bleu_n(list_of_references, list_of_hypothesis, 4)} ')
    print(f'Meteor Score is: {mean(meteors)}')
    results = bertscore.compute(predictions = list_of_hypothesis, references = list_of_references, lang="en")
    # b_p = results['precision']
    b_f1 = results['f1']
    # b_r = results['recall']
    # print(f'Bert Score R is: {mean(b_r)}')
    # print(f'Bert Score P is: {mean(b_p)}')
    print(f'Bert Score F is: {mean(b_f1)}')
    print(f'Rouge R is: {average(rouge_r)}')
    print(f'Rouge P is: {average(rouge_p)}')
    print(f'Rouge F is: {average(rouge_f)}')
    print(f"ROC AUC score is: {roc_auc_score(all_labels, soft_max, multi_class = 'ovo')}")
    df = pd.DataFrame(list(zip(gen_words, ground_words)), columns = ['generated', 'ground'])
    df.to_csv('outputs.csv')

def finetune_train(train_loader, device, llm_model, optimizer, total_num):
    llm_model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc='- (Training)  '): 

        sig, report, label, input_ids = map(lambda x: x.to(device), batch[:-1])
        optimizer.zero_grad()
        input_ids = input_ids[:, -1, :]
        pred = llm_model(input_ids = input_ids, labels = input_ids)
        loss = pred.loss
        logits = pred.logits        
        loss.backward()
        optimizer.step_and_update_lr()

        total_loss += loss.item()

    train_loss = total_loss / total_num
    return train_loss

def finetune_validate(valid_loader, device, llm_model, total_num, tokenizer):
    bleu = []
    rouge_r =[]
    rouge_p =[]
    rouge_f =[]
    llm_model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc='- (Validation)  '):
            sig, report, label, input_ids = map(lambda x: x.to(device), batch[:-1])
            words = batch[-1]
            input_ids = input_ids[:, -1, :]
            pred = llm_model(input_ids = input_ids, labels = input_ids)
            loss = pred.loss

            total_loss += loss.item()

            b_scores = []
            r_scores = []
            p_scores = []
            f_scores = []
        
            generated_ids = llm_model.generate(input_ids)
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens = True)
	    
            for idx in range(len(generated_text)):
                try:
                    print('Generated: ', generated_text[idx])
                    print('Ground True: ', words[idx])
                    b_score = calc_bleu_n(generated_text[idx], words[idx], 1)
                    r, p, f = calc_rouge(generated_text[idx], words[idx])
                    b_scores.append(b_score)
                    r_scores.append(r)
                    f_scores.append(f)
                    p_scores.append(p)
                except ValueError:
                    pass
            rouge_r.append(average(r_scores))
            rouge_f.append(average(f_scores))
            rouge_p.append(average(p_scores))
            bleu.append(average(b_scores))

    print(f'Bleu Score is: {average(bleu)} ')
    print(f'Rouge R is: {average(rouge_r)}')
    print(f'Rouge P is: {average(rouge_p)}')
    print(f'Rouge F is: {average(rouge_f)}')
    valid_loss = total_loss / total_num
    print(f'Loss: {valid_loss}')
    return valid_loss

def finetune_test(test_loader, device, llm_model, total_num, tokenizer):
    bleu = []
    rouge_r =[]
    rouge_p =[]
    rouge_f =[]
    llm_model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='- (Testing)  '):
            sig, report, label, input_ids = map(lambda x: x.to(device), batch[:-1])
            words = batch[-1]
            input_ids = input_ids[:, -1, :]
            pred = llm_model(input_ids = input_ids, labels = input_ids)
            loss = pred.loss
            b_scores = []
            r_scores = []
            p_scores = []
            f_scores = []
            generated_ids = llm_model.generate(input_ids)
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens = True)
            for idx in range(len(generated_text)):
                try:
                    print('Generated: ', generated_text[idx])
                    print('Ground True: ', words[idx])
                    b_score = calc_bleu_n(generated_text[idx], words[idx], 1)
                    r, p, f = calc_rouge(generated_text[idx], words[idx])
                    b_scores.append(b_score)
                    r_scores.append(r)
                    f_scores.append(f)
                    p_scores.append(p)
                except ValueError:
                    pass
            rouge_r.append(average(r_scores))
            rouge_f.append(average(f_scores))
            rouge_p.append(average(p_scores))
            bleu.append(average(b_scores))

            total_loss += loss.item()
    test_loss = total_loss / total_num
    print(f'Bleu Score is: {average(bleu)} ')
    print(f'Rouge R is: {average(rouge_r)}')
    print(f'Rouge P is: {average(rouge_p)}')
    print(f'Rouge F is: {average(rouge_f)}')
    print(f'Loss is: {test_loss}')

def cal_statistic(cm):
    total_pred = cm.sum(0)
    total_true = cm.sum(1)

    acc_SP = sum([cm[i, i] for i in range(1, class_num)]) / total_pred[1: class_num].sum()
    pre_i = [cm[i, i] / total_pred[i] for i in range(class_num)]
    rec_i = [cm[i, i] / total_true[i] for i in range(class_num)]
    F1_i = [2 * pre_i[i] * rec_i[i] / (pre_i[i] + rec_i[i]) for i in range(class_num)]

    pre_i = np.array(pre_i)
    rec_i = np.array(rec_i)
    F1_i = np.array(F1_i)
    pre_i[np.isnan(pre_i)] = 0
    rec_i[np.isnan(rec_i)] = 0
    F1_i[np.isnan(F1_i)] = 0

    return acc_SP, average(list(pre_i)), average(list(rec_i)), average(list(F1_i))

def average(lst):
    return sum(lst) / len(lst)

def sentence_calc_bleu_n(pred, label, n):
    if n == 1:
        score = nltk.translate.bleu_score.sentence_bleu(label, pred, weights=(1, 0, 0, 0))
    elif n == 2:
        score = nltk.translate.bleu_score.sentence_bleu(label, pred, weights = (0.5, 0.5, 0, 0))
    elif n == 3:
        score = nltk.translate.bleu_score.sentence_bleu(label, pred, weights = (0.33, 0.33, 0.33, 0))
    elif n == 4:
        score =nltk.translate.bleu_score.sentence_bleu(label, pred, weights = (0.25, 0.25, 0.25, 0.25))
    return score

def calc_bleu_n(pred, label, n):
    if n == 1:
        score = nltk.translate.bleu_score.corpus_bleu(label, pred, weights=(1, 0, 0, 0))
    elif n == 2:
        score = nltk.translate.bleu_score.corpus_bleu(label, pred, weights = (0.5, 0.5, 0, 0))
    elif n == 3:
        score = nltk.translate.bleu_score.corpus_bleu(label, pred, weights = (0.33, 0.33, 0.33, 0))
    elif n == 4:
        score =nltk.translate.bleu_score.corpus_bleu(label, pred, weights = (0.25, 0.25, 0.25, 0.25))
    return score

def calc_rouge(pred, label):
    rouge = Rouge()
    score = rouge.get_scores(pred, label)
    score = score[0]['rouge-1']
    r = score['r']
    p = score['p']
    f = score['f']
    
    
    return r, p, f

def translate(model, list_of_sentences):
    translated_sentences = model.translate(list_of_sentences, target_lang='en', source_lang = 'de', show_progress_bar = True)
    translated_sentences = np.array(translated_sentences)

    return translated_sentences
    
def get_embeddings(df, model, tokenizer, device):
    tokenized_words = []
    hidden_states = []
    embeddings = []
    input_ids = []
    words = []
    for word in df:
        words.append(word)
        inputs = tokenizer(word, max_length = 30, padding = 'max_length', return_tensors="pt", truncation = True)
        input_ids.append(inputs.input_ids)
        tokenized_words.append(inputs)
    for input in tqdm(tokenized_words, desc = 'Getting Embeddings: '):
        with torch.no_grad():
            input = input.to(device)
            output = model(**input)
            hidden_state = output[2][0].detach()
            embedding = torch.mean(hidden_state, dim=1)
            embeddings.append(embedding.cpu().detach())

    return embeddings, input_ids, words
        
    
def preprocess(X, y):
    new_X = []
    new_y = []
    for i in range(len(y)):
        if ('Ö' in y[i,0]) or ('Å' in y[i, 0]) or ('ö' in y[i, 0]) or ('ä' in y[i, 0]):
            pass
        elif ('4.46' in y[i, 0]):
            split_text = y[i, 0].split(' ')
            idx = split_text.index('4.46')
            split_text = split_text[:idx]
            joined_text = ' '.join(split_text)
            new_X.append(X[i, :].tolist())
            replaceed = [joined_text, y[i, 1]]
            new_y.append(replaceed)
        else:
            same = [y[i, 0], y[i, 1]]
            new_y.append(same)
            new_X.append(X[i, :].tolist())


    new_array_X = np.array(new_X)
    new_array_y = np.array(new_y)
    return new_array_X, new_array_y
