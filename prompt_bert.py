from transformers import BertForMaskedLM, BertConfig, BertTokenizer, RobertaTokenizer, RobertaConfig, RobertaForMaskedLM
from torch import nn
import torch
import torch.nn.functional as F


class PromptBertModel(nn.Module):
    def __init__(self, args):
        super(PromptBertModel, self).__init__()
        self.config = BertConfig.from_pretrained('bert-large-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        self.model = BertForMaskedLM.from_pretrained('bert-large-uncased')
        # self.config = RobertaConfig.from_pretrained('roberta-large')
        # self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        # self.model = RobertaForMaskedLM.from_pretrained('roberta-large')

        self.prefix_soft_index, self.suffix_soft_index = [3, 27569, 10], [11167, 10]
        self.p_num, self.s_num = len(self.prefix_soft_index), len(self.suffix_soft_index)
        self.prefix_soft_embedding_layer = nn.Embedding(
            self.p_num, self.config.hidden_size
        )
        self.suffix_soft_embedding_layer = nn.Embedding(
            self.s_num, self.config.hidden_size
        )
        self.normal_embedding_layer = self.model.get_input_embeddings()
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
        self.mask_ids = torch.tensor([self.tokenizer.mask_token_id])
        for param in self.model.parameters():
            param.requires_grad_(False)

    def forward(self, input_ids, attention_mask, labels=None):
        batch_size = input_ids.shape[0]
        prefix_soft_ids = torch.stack([self.prefix_soft_ids for i in range(batch_size)]).to(input_ids.device)
        mask_ids = torch.stack([self.mask_ids for i in range(batch_size)]).to(input_ids.device)
        suffix_soft_ids = torch.stack([self.suffix_soft_ids for i in range(batch_size)]).to(input_ids.device)

        prefix_soft_embeddings = self.prefix_soft_embedding_layer(prefix_soft_ids)
        suffix_soft_embeddings = self.suffix_soft_embedding_layer(suffix_soft_ids)

        text_embeddings = self.normal_embedding_layer(input_ids)
        mask_embeddings = self.normal_embedding_layer(mask_ids)
        input_embeddings = torch.cat(
            [prefix_soft_embeddings, text_embeddings, suffix_soft_embeddings, mask_embeddings],
            dim=1
        )
        prefix_soft_attention_mask = torch.ones(batch_size, self.p_num).to(input_ids.device)
        mask_attention_mask = torch.ones(batch_size, 1).to(input_ids.device)
        suffix_soft_attention_mask = torch.ones(batch_size, self.s_num).to(input_ids.device)

        attention_mask = torch.cat(
            [prefix_soft_attention_mask, attention_mask, suffix_soft_attention_mask, mask_attention_mask],
            dim=1
        )
        outputs = self.model(inputs_embeds=input_embeddings, attention_mask=attention_mask)[0]
        # masked_token_pos = torch.full(masked_token_pos.shape, 50 + self.p_num).to(input_ids.device)
        # vocab_size = outputs.shape[2]
        # masked_token_pos = torch.unsqueeze(masked_token_pos, 1)
        # masked_token_pos = torch.unsqueeze(masked_token_pos, 2)
        # masked_token_pos = torch.stack([masked_token_pos] * vocab_size, 2)
        # masked_token_pos = torch.squeeze(masked_token_pos, 3)
        # masked_token_logits = torch.gather(outputs, 1, masked_token_pos)
        #
        # masked_token_logits = masked_token_logits.reshape(-1, vocab_size)
        logits = outputs[:, -1, [8699, 4963, 12721, 3571, 6569, 12039, 4474]]

        probs = F.log_softmax(logits, -1)
        if labels != None:
            loss = F.nll_loss(probs, labels.to(probs.device), reduction='mean')
            return loss, logits
        else:
            return logits