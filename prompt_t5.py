from transformers import T5ForConditionalGeneration, T5Config
from torch import nn
import torch
import torch.nn.functional as F


class PromptT5CLSModel(nn.Module):
    def __init__(self, args):
        super(PromptT5CLSModel, self).__init__()
        self.config = T5Config.from_pretrained('t5-base')
        self.t5 = T5ForConditionalGeneration.from_pretrained('t5-base')
        self.soft_embedding_layer = None
        self.normal_embedding_layer = self.t5.get_input_embeddings()
        # self.proj_linear = nn.Linear(22, 7)

        # self.prefix_soft_index, self.suffix_soft_index = [3, 27569, 10], [31484, 17, 10, 1]
        # self.prefix_soft_index, self.suffix_soft_index = [8, 6493, 13], [19, 1]
        self.prefix_soft_index, self.suffix_soft_index = [8, 6493, 13], [31484, 17, 10, 1]
        self.p_num, self.s_num = len(self.prefix_soft_index), len(self.suffix_soft_index)
        self.prefix_soft_embedding_layer = nn.Embedding(
            self.p_num, self.config.hidden_size
        )
        self.suffix_soft_embedding_layer = nn.Embedding(
            self.s_num, self.config.hidden_size
        )
        self.prefix_soft_embedding_layer.weight.data = torch.stack(
            [self.normal_embedding_layer.weight.data[i, :].clone().detach().requires_grad_(True) for i in
             self.prefix_soft_index]
        )
        self.suffix_soft_embedding_layer.weight.data = torch.stack(
            [self.normal_embedding_layer.weight.data[i, :].clone().detach().requires_grad_(True) for i in
             self.suffix_soft_index]
        )
        self.prefix_soft_ids = torch.tensor(range(self.p_num))
        self.suffix_soft_ids = torch.tensor(range(self.s_num))
        for param in self.t5.parameters():
            param.requires_grad_(False)

    def forward(self, input_ids, attention_mask, labels=None):
        batch_size = input_ids.shape[0]
        decoder_input_ids = torch.zeros(batch_size, 1, dtype=int).to(input_ids.device)
        prefix_soft_ids = torch.stack([self.prefix_soft_ids for i in range(batch_size)]).to(input_ids.device)
        suffix_soft_ids = torch.stack([self.suffix_soft_ids for i in range(batch_size)]).to(input_ids.device)

        prefix_soft_embeddings = self.prefix_soft_embedding_layer(prefix_soft_ids)
        suffix_soft_embeddings = self.suffix_soft_embedding_layer(suffix_soft_ids)

        text_embeddings = self.normal_embedding_layer(input_ids)

        input_embeddings = torch.cat(
            [prefix_soft_embeddings, text_embeddings, suffix_soft_embeddings],
            dim=1
        )

        prefix_soft_attention_mask = torch.ones(batch_size, self.p_num).to(input_ids.device)
        suffix_soft_attention_mask = torch.ones(batch_size, self.s_num).to(input_ids.device)
        attention_mask = torch.cat(
            [prefix_soft_attention_mask, attention_mask, suffix_soft_attention_mask],
            dim=1
        )
        output = self.t5(
            inputs_embeds=input_embeddings,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        logits = output['logits'][:, 0, [7163, 11213, 27635, 2971, 3922, 24784, 4158]]
        # logits = output['logits'][:, 0, [7163, 16, 25880, 11213, 1080, 12603, 27635, 13006, 5591,
        #                                  2971, 7403, 6541, 15, 3922, 1095, 5010, 24784, 26887,
        #                                  10875, 4158, 12914, 7544]]
        # logits = self.proj_linear(logits)
        probs = F.log_softmax(logits, -1)
        if labels != None:
            loss = F.nll_loss(probs, labels.to(probs.device), reduction='mean')
            return loss, logits
        else:
            return logits
