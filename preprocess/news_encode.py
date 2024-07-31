import argparse
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import json
import pickle
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch

model_name = '../deberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
module = AutoModel.from_pretrained(model_name, output_hidden_states=True)
#device = 'cuda:0'
module.eval()
#module.to(device)
module.cuda()

def news_encoder(des):
    des_token = tokenizer.encode(des, add_special_tokens=True)[:256]
    input_ids = torch.tensor(des_token).unsqueeze(dim=0).cuda()
    with torch.no_grad():
        outputs = module(input_ids)
        #svec0 = outputs[-1]
        #svec1 = svec0[-1]
        svec = outputs['last_hidden_state']
        svec_np = svec.detach().cpu().numpy()
        svec_np = svec_np[0]
    return svec_np[0]

def train_news_encoder(source_folder, target_folder, dataset_list):
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    for dataset in dataset_list:
        with open(os.path.join(source_folder, dataset), 'r', encoding='utf-8') as f:

            train_news_list = json.load(f)
        svecs_list = list()
        for index, train_news in enumerate(tqdm(train_news_list, total=len(train_news_list))):

            des = ' '.join(train_news['keywords'])
            des = des.strip()
            svec = news_encoder(des)
            svecs_list.append(svec)
        svecs_list = np.asarray(svecs_list)
        np.save(os.path.join(target_folder, 'encoded_' + dataset), svecs_list)
        print(os.path.join(target_folder, 'encoded_' + dataset) + ' done')
    print('finish')


if __name__ == '__main__':
    start2 = time.time()
    source_folder = '../dataset_full'
    target_folder = '../dataset_full/encode'

    dataset_list = ['selected_valid_gossipcop.jsonl', 'selected_train_gossipcop.jsonl', 'gossipcop_test.jsonl']
    news_t_list = ['news_t.jsonl']
    train_news_encoder(source_folder, target_folder, news_t_list)
    end2 = time.time()
    with open('time_encode.txt', 'w') as t:
        t.writelines('time_encode: {} seconds'.format(end2 - start2))
