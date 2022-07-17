import torch
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from transformers import logging
from bert_model import CLSModel
from t5_model import T5CLSModel
import argparse

class SCDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_size = len(dataset)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        return self.dataset[index]


#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
logging.set_verbosity_error()

def coffate_fn_test(examples):
    inputs, targets = [], []
    for sent in examples:
        inputs.append(sent)
        targets.append(-1)
    inputs = tokenizer(inputs,
                       padding=True,
                       truncation=True,
                       return_tensors="pt",
                       max_length=512)
    targets = torch.tensor(targets)
    return inputs, targets

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--test_data_path",
        default='./data/mydata/dev.json',
        type=str,
        help="The testing data path.",
    )
    parser.add_argument(
        "--test_label_path",
        default='./data/mydata/dev_labels.json',
        type=str,
        help="The testing label path.",
    )
    parser.add_argument(
        "--max_len",
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--test_batch_size",
        default=128,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--hidden_size",
        default=768,
        type=int,
        help="Hidden size.",
    )
    parser.add_argument(
        "--project_dim",
        default=7,
        type=int,
        help="Project Dim.",
    )
    parser.add_argument(
        "--bert_checkpoint_dir",
        default='./checkpoint/save_model_best.pt',
        type=str,
        help="the path of fine-tuned bert checkpoint.",
    )
    parser.add_argument(
        "--t5_checkpoint_dir",
        default='./checkpoint/save_t5_model_best.pt',
        type=str,
        help="the path of fine-tuned t5 checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        default='./checkpoint',
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="random seed for initialization",
    )
    args = parser.parse_args()
    test_data_path = "./data/mydata/test.tsv"
    test_data = []
    with open(test_data_path, 'r', encoding="utf-8") as fr:
        for line in fr.readlines():
            sentence = line.strip()
            test_data.append(sentence)
    test_dataset = SCDataset(test_data)
    test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             collate_fn=coffate_fn_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = CLSModel(args)
    model.load_state_dict(torch.load(args.bert_checkpoint_dir)['model'])
    model.cuda()

    res = []
    for batch in tqdm(test_dataloader, desc=f"Testing"):
        inputs, targets = [x.to(device) for x in batch]
        with torch.no_grad():
            bert_output = model(inputs)
            #print(bert_output.argmax(dim=1).data.item())
            res.append(str(bert_output.argmax(dim=1).data.item()))

    with open('test_res.tsv', 'w', encoding='utf-8') as fw:
        for item in res:
            fw.write(item+'\n')