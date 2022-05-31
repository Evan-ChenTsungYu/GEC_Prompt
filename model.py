
from turtle import forward
import torch.nn as nn
import torch
import copy
from transformers import T5Config, T5ForConditionalGeneration, T5Model


class Prompt_Model(nn.Module):
    def __init__(self, pretrain_name):
        super().__init__()
        config = T5Config.from_pretrained(pretrain_name)
        self.prompt_model = T5ForConditionalGeneration.from_pretrained(pretrain_name, config = config )
        self.linear = nn.Linear(5,5)

    def forward(self, wrong_sent,wrong_sent_mask, correct_sent):
        wrong_sent_output = self.prompt_model(input_ids = wrong_sent, attention_mask = wrong_sent_mask , labels = correct_sent) #dimention [b,s,768]
        return wrong_sent_output

    def generate(self, wrong_sent, max_len):
        return self.prompt_model.generate(wrong_sent, num_beams = 8, max_length = max_len)

# if __name__ == '__main__':
    