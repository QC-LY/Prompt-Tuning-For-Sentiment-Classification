import os
import logging
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import torch

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def get_metric(preds, golds):

    label_list = ['neutral', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    golds = [label_list[gold] for gold in golds]
    preds = [label_list[pred] for pred in preds]
    acc = accuracy_score(golds, preds)
    f1_macro = f1_score(golds, preds, average='macro')

    return {'acc': acc, 'f1': f1_macro}

def eval_model(model, valid_dataloader):
    model.eval()
    predict_list = []
    golden_list = []
    with torch.no_grad():
        for batch in tqdm(valid_dataloader):
            input_ids = batch[0].cuda()
            attention_mask = batch[1].cuda()
            labels = batch[2]
            outputs = model(input_ids, attention_mask)
            max_score, max_idxs = torch.max(outputs, 1)
            predict_idxs = max_idxs.view(-1).tolist()
            predict_list.extend(predict_idxs)
            golden_idxs = labels.view(-1).tolist()
            golden_list.extend(golden_idxs)
        evaluation_results = get_metric(predict_list, golden_list)
        return evaluation_results


def set_env(args, run_type):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    handlers = [logging.FileHandler(os.path.abspath(args.output_dir) + '/' + run_type + '_log.txt'), logging.StreamHandler()]
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=handlers)
    set_seed(args)