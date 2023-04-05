from transformers import BertTokenizer, T5Tokenizer
import os
import argparse
import logging
import torch
from torch.utils.data import DataLoader
from data_ultis import ScData
from bert_model import CLSModel
from t5_model import T5CLSModel
from utils import set_env, eval_model
from tqdm import tqdm

logger = logging.getLogger(__name__)


def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--test_data_path",
        default='./data/test.json',
        type=str,
        help="The testing data path.",
    )
    parser.add_argument(
        "--test_label_path",
        default='./data/test_labels.json',
        type=str,
        help="The testing label path.",
    )
    parser.add_argument(
        "--max_len",
        default=150,
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
    log_name = args.model_name + '_test'
    set_env(args, log_name)
    if args.model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = CLSModel(args)
        model.load_state_dict(torch.load(args.bert_checkpoint_dir)['model'])
        model.cuda()
    else:
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        model = T5CLSModel(args)
        # model.load_state_dict(torch.load(args.t5_checkpoint_dir)['model'])
        model.cuda()
    test_data = ScData(args, args.test_data_path, args.test_label_path, tokenizer)
    test_reader = DataLoader(dataset=test_data, num_workers=0,
                             batch_size=args.test_batch_size, shuffle=False)

    res = []
    with torch.no_grad():
        for batch in tqdm(test_reader):
            input_ids = batch[0].cuda()
            attention_mask = batch[1].cuda()
            outputs = model(input_ids, attention_mask)
            max_score, max_idxs = torch.max(outputs, 1)
            predict_idxs = max_idxs.view(-1).tolist()
            res.extend(predict_idxs)
    # result = eval_model(model, test_reader)
    # logger.info('test acc: {0}, F1: {1}'.format(result['acc'], result['f1']))
    print('length of test:', len(res))
    with open('output/20195199_3.csv', 'w', encoding='utf-8') as fw:
        for item in res:
            fw.write(str(item)+'\n')
    print('saved!')


if __name__ == "__main__":
    inference()
