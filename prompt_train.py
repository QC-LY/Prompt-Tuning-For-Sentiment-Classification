import sys
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    BertTokenizer,
    T5Tokenizer,
    RobertaTokenizer
)
import os
import argparse
import logging
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, DistributedSampler
from data_ultis import ScData
from prompt_bert import PromptBertModel
from prompt_t5 import PromptT5CLSModel
from utils import set_env, set_seed, get_metric, eval_model

logger = logging.getLogger(__name__)


def train(args, model, train_dataloader, valid_dataloader):
    real_batch_size = args.train_batch_size * args.gradient_accumulation_steps
    t_total = train_dataloader.dataset.__len__() // real_batch_size * args.num_train_epochs

    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer], 'weight_decay': 0.01}]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    global_step = 0
    tr_loss = 0.0
    best_acc = 0.0
    model.zero_grad()
    for epoch in range(int(args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            model.train()
            input_ids = batch[0].cuda()
            attention_mask = batch[1].cuda()
            labels = batch[2].cuda()
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs[0]
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                logger.info(
                    'Epoch: {}, Step: {}, Loss: {:.4f}, lr: {:.6f}'.format(epoch, global_step, (tr_loss / global_step),
                                                                           optimizer.param_groups[0]["lr"]))
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                if global_step % args.eval_steps == 0:
                    logger.info('Start eval!')
                    evaluation_results = eval_model(model, valid_dataloader)
                    acc = evaluation_results["acc"]
                    f1 = evaluation_results["f1"]
                    logger.info('Dev acc: {0}, F1: {1}'.format(acc, f1))
                    if acc >= best_acc:
                        best_acc = acc
                        if args.model_name_or_path == 'bert':
                            torch.save({'epoch': epoch,
                                        'model': model.state_dict()},
                                       os.path.join(args.output_dir, "save_prompt_bert_model_best.pt"))
                        else:
                            torch.save({'epoch': epoch,
                                        'model': model.state_dict()},
                                       os.path.join(args.output_dir, "save_prompt_t5_model_best.pt"))
                        logger.info("Saved best epoch {0}, best acc {1}".format(epoch, best_acc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_path",
        default=None,
        type=str,
        required=True,
        help="The training data path.",
    )
    parser.add_argument(
        "--dev_data_path",
        default=None,
        type=str,
        required=True,
        help="The validation data path.",
    )
    parser.add_argument(
        "--train_label_path",
        default=None,
        type=str,
        required=True,
        help="The training label path.",
    )
    parser.add_argument(
        "--dev_label_path",
        default=None,
        type=str,
        required=True,
        help="The validation label path.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_len",
        default=150,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--train_batch_size",
        default=128,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--dev_batch_size",
        default=128,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.",
    )
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.",
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm.",
    )
    parser.add_argument(
        "--bitfit",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=10,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps.",
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
        "--eval_steps",
        type=int,
        default=500,
        help="eval model every X updates steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="random seed for initialization",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        type=str
    )
    parser.add_argument(
        "--suffix",
        default=None,
        type=str
    )
    args = parser.parse_args()
    log_name = args.model_name_or_path + '_train_prompt'
    set_env(args, log_name)
    if args.model_name_or_path == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        # tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        model = PromptBertModel(args)
    else:
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        model = PromptT5CLSModel(args)
    model.cuda()
    logger.info("Loading training set.")
    train_data = ScData(args, args.train_data_path, args.train_label_path, tokenizer)
    train_sampler = RandomSampler(train_data)
    train_reader = DataLoader(dataset=train_data, sampler=train_sampler, num_workers=0,
                              batch_size=args.train_batch_size)
    dev_data = ScData(args, args.dev_data_path, args.dev_label_path, tokenizer)
    dev_reader = DataLoader(dataset=dev_data, num_workers=0,
                            batch_size=args.dev_batch_size, shuffle=False)
    train(args, model, train_reader, dev_reader)


if __name__ == "__main__":
    main()
