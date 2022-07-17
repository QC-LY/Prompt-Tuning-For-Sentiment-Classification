from transformers import T5ForConditionalGeneration
from torch import nn
import torch
import torch.nn.functional as F


class T5CLSModel(nn.Module):
    def __init__(self, args):
        super(T5CLSModel, self).__init__()
        # self.encode_proj = nn.Linear(args.hidden_size, args.project_dim)
        self.model = T5ForConditionalGeneration.from_pretrained('t5-base')

    def forward(self, input_ids, attention_mask, labels=None):
        batch_size = input_ids.shape[0]
        decoder_input_ids = torch.zeros(batch_size, 1, dtype=int).to(input_ids.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             decoder_input_ids=decoder_input_ids, return_dict=True)
        logits = outputs['logits'][:, 0, [7163, 11213, 27635, 2971, 3922, 24784, 4158]]
        probs = F.log_softmax(logits, -1)
        if labels != None:
            loss = F.nll_loss(probs, labels.to(probs.device), reduction='mean')
            return loss, logits
        else:
            return logits
