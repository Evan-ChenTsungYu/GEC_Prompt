import argparse
import torch
import json
from  make_dataset import load_GEC_dataset, GEC_DS
from model import Prompt_Model
from transformers import T5ForConditionalGeneration, T5Tokenizer
from evaluate import evaluate
from train import train
from parameter import param 
import torch.nn as nn
import os

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-tti',  type = str, help='input train/test', dest = "TT") #現在要train/test/inference
    parser.add_argument('-e', type = int, help = 'epoch to train', dest="epoch")
    parser.add_argument('-g', type = int, help = 'Which GPU to used', dest = 'GPU')
    args, remaining_args = parser.parse_known_args() #args 是存已知的args, remain 是存使用者不小心輸錯的
    
    DEVICE = 'cuda:'+str(args.GPU)
    # DEVICE = torch.device("cuda:1,3")
    parameter = param()

    tokenizer = T5Tokenizer.from_pretrained(parameter.model_name)
    tokenizer.add_tokens(['[', '|', ']', 'NONE'])
    print(len(tokenizer))
    dataset_list = {'train': ['lang8'], 'test':['conll'], 'valid':['wi', 'fce']}
    TT_DL, Val_DL = load_GEC_dataset(dataset_list, args.TT , parameter.batch_size, parameter.split_size, parameter.val_split_size) #load_dataset with train/test/inference
    
    Model = Prompt_Model('t5-base').to(DEVICE)
    Model.prompt_model.resize_token_embeddings(len(tokenizer))
    Model.train()
    if args.TT == 'train':
        print('start Prompt training')
        train(Model, TT_DL, Val_DL, args.epoch, parameter.result_save_path,tokenizer, show_epoch_result = True, DEVICE = DEVICE)
    else:
        print(f'start testing on {dataset_list[args.TT]} with {parameter.batch_size*len(TT_DL)} sentence')
        param_dict = parameter.parameter_dict()
        Model.load_state_dict(torch.load(parameter.result_save_path, map_location = DEVICE))
        Model.eval()
        evaluate(Model, TT_DL, parameter.max_len, DEVICE, tokenizer)
        
        # result_json(result, param_dict, './result'+'.json')
    return 0


if __name__ == "__main__" :
    main()