# -*- coding: utf-8 -*-

# libraries
import matplotlib.pyplot as plt
import pandas as pd
import torch
import seaborn as sns

# Preliminarires
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator

# Models
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

# Training
import torch.optim as optim


# Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

## Process dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Model parameter
MAX_SEQ_LEN = 126
PAD_INDEX =tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX =tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

# Fields 
label_field = Field(sequential=False, use_vocab=False,batch_first=True,dtype=torch.int32)
text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False,batch_first=True,
                   fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
fields = [('label',label_field),('response',text_field),('context',text_field),('res_cont',text_field)]



train = TabularDataset(path='train.csv', format = 'CSV', fields=fields,skip_header=True)
valid = TabularDataset(path='dev.csv', format = 'CSV', fields=fields,skip_header=True)
test = TabularDataset(path='dev.csv', format = 'CSV', fields=fields,skip_header=True)


# Iterators
device = torch.device('cuda')

train_iter = BucketIterator(train,device=torch.device('cuda'),batch_size=16,sort_key=lambda x:len(x.response),train=True, 
               sort=True, sort_within_batch=True)
valid_iter = BucketIterator(valid,device=torch.device('cuda'),batch_size=16,sort_key=lambda x:len(x.response),train=True, 
               sort=True, sort_within_batch=True)
test_iter = Iterator(test,batch_size=16,device=torch.device('cuda'),train=False,shuffle=False,sort=False)


class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name)

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]

        return loss, text_fea
    
    
# Save and Load Functions

def save_checkpoint(save_path, model, valid_loss):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model):
    
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


# Training Function

def train(model,
          optimizer,
          criterion = nn.BCELoss(),
          train_loader = train_iter,
          valid_loader = valid_iter,
          num_epochs = 5,
          eval_every = len(train_iter) // 2,
          file_path = 'result',
          best_valid_loss = float("Inf")):
    
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        for (labels, response, context, res_cont), _ in train_loader:
            labels = labels.type(torch.LongTensor)           
            labels = labels.to(device)
            res_cont = res_cont.type(torch.LongTensor)  
            res_cont = res_cont.to(device)
            output = model(res_cont, labels)
            loss, _ = output

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():                    

                    # validation loop
                    for (labels, response, context, res_cont), _ in valid_loader:
                        labels = labels.type(torch.LongTensor)           
                        labels = labels.to(device)
                        res_cont = res_cont.type(torch.LongTensor)  
                        res_cont = res_cont.to(device)
                        output = model(res_cont, labels)
                        loss, _ = output
                        
                        valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_valid_loss))
                
                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/' + 'model.pt', model, best_valid_loss)
                    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    
    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')

model = BERT().to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-5)

train(model=model, optimizer=optimizer)


train_loss_list, valid_loss_list, global_steps_list = load_metrics('result' + '/metrics.pt')
plt.plot(global_steps_list, train_loss_list, label='Train')
plt.plot(global_steps_list, valid_loss_list, label='Valid')
plt.xlabel('Global Steps')
plt.ylabel('Loss')
plt.legend()
plt.show() 


def evaluate(model, test_loader):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for (labels, response, context, res_cont), _ in test_loader:

                labels = labels.type(torch.LongTensor)           
                labels = labels.to(device)
                res_cont = res_cont.type(torch.LongTensor)  
                res_cont = res_cont.to(device)
                output = model(res_cont, labels)

                _, output = output
                y_pred.extend(torch.argmax(output, 1).tolist())
                y_true.extend(labels.tolist())
    
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['NOT_SARCASM', 'SARCASM'])
    ax.yaxis.set_ticklabels(['NOT_SARCASM', 'SARCASM'])
    
best_model = BERT().to(device)

load_checkpoint('result' + '/model.pt', best_model)

evaluate(best_model, test_iter)



#%% final result
def test(model, test_loader):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for (labels, response, context, res_cont), _ in test_loader:

                labels = labels.type(torch.LongTensor)           
                labels = labels.to(device)
                res_cont = res_cont.type(torch.LongTensor)  
                res_cont = res_cont.to(device)
                output = model(res_cont, labels)

                _, output = output
                y_pred.extend(torch.argmax(output, 1).tolist())
                y_true.extend(labels.tolist())
    
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['NOT_SARCASM', 'SARCASM'])
    ax.yaxis.set_ticklabels(['NOT_SARCASM', 'SARCASM'])
    
    return y_pred


testData = TabularDataset(path='testData.csv', format = 'CSV', fields=fields,skip_header=True)
testData_iter = Iterator(testData,batch_size=16,device=torch.device('cuda'),train=False,shuffle=False,sort=False)
result = test(best_model, testData_iter)

#%% Export result data
test_raw = pd.read_json('test.jsonl', orient = 'columns', lines = True)
test_ID = test_raw[['id']]
y_pred = pd.Series(result)
y_pred.replace(1, "SARCASM", inplace=True)
y_pred.replace(0, 'NOT_SARCASM', inplace = True)
test_ID['prediction'] = y_pred
test_ID.to_csv(path_or_buf = 'result/answer.txt', index = False, header = False)