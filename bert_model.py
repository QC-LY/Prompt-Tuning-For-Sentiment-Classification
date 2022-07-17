from transformers import BertModel, RobertaModel
from torch import nn
import torch
import torch.nn.functional as F


class CLSModel(nn.Module):
    def __init__(self, args):
        super(CLSModel, self).__init__()
        self.encode_proj = nn.Linear(args.hidden_size, args.project_dim)
        self.model = BertModel.from_pretrained('bert-base-uncased')
        # self.model = RobertaModel.from_pretrained("roberta-base")

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = outputs.pooler_output
        logits = self.encode_proj(pooler_output)
        probs = F.log_softmax(logits, -1)
        if labels != None:
            loss = F.nll_loss(probs, labels.to(probs.device), reduction='mean')
            return loss, logits
        else:
            return logits