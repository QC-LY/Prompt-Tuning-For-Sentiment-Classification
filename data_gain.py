import json
import requests
from tqdm import tqdm

url = 'http://api.niutrans.com/NiuTransServer/translation?'

sents = json.load(open('data/mydata/train.json'))
lable = json.load(open('data/mydata/train_labels.json'))
# print(a)
new_train = []
new_train_label = []
for i in tqdm(range(len(sents))):
    if lable[i] == 0:
        new_train.append(sents[i])
        new_train_label.append(lable[i])
    else:
        new_train.append(sents[i])
        new_train_label.append(lable[i])
        data = {"from": 'en', "to": 'zh', "apikey": '949926ff411994c511daca5f30c83a02', "src_text": sents[i]}
        res = requests.post(url, data=data).json()
        new_data = {"from": 'zh', "to": 'en', "apikey": '949926ff411994c511daca5f30c83a02', "src_text": str(res['tgt_text'])}
        new_res = requests.post(url, data=new_data).json()
        new_sent = new_res['tgt_text']
        new_train.append(new_sent)
        new_train_label.append(lable[i])
# json.dump(new_train, open('data/mydata/new_train.json', 'w'))
# json.dump(new_train_label, open('data/mydata/new_train_labels.json', 'w'))
# a = json.load(open('data/mydata/new_train.json', 'r'))
# print(len(a))