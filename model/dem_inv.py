import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
import time
import json
from LLM_detected_and_save import LLM_save
from prompt_news_select import prompt_news_select
import numpy as np
import argparse
import random

def train_data_shot(shot, train_data):
    fake_num = shot
    real_num = shot
    train_news_index_list = []
    for index, news in enumerate(train_data):
        if news['label'] == 'fake' and fake_num > 0:
            fake_num -= 1
            train_news_index_list.append(index)
        elif news['label'] == 'real' and real_num > 0:
            real_num -= 1
            train_news_index_list.append(index)
    return train_news_index_list
def extract_train_news_and_ask_LLM(shot, test_news_dataset, source_folder,
                                   encode_folder, target_folder, output_folder):
    news_jsonl_files = [file for file in os.listdir(source_folder) if test_news_dataset in file]
    for news_jsonl in news_jsonl_files:
        if 'test' in news_jsonl:
            with open(os.path.join(source_folder, news_jsonl), 'r') as f:
                test_news_list = json.load(f)
        elif 'train' in news_jsonl:
            with open(os.path.join(source_folder, news_jsonl), 'r') as f:
                train_data_list = json.load(f)
        elif 'news_t' in news_jsonl:
            with open(os.path.join(source_folder, news_jsonl), 'r') as f:
                test_news_list = json.load(f)
            with open(os.path.join(source_folder, "selected_train_politifact.jsonl"), 'r') as f:
                train_data_list = json.load(f)

    train_index_list = train_data_shot(shot, train_data_list)
    train_news_list = [train_data_list[i] for i in train_index_list]

    train_encode_list = []
    test_news_encode_list = []
    for encoded_file in os.listdir(encode_folder):
        if test_news_dataset in encoded_file:
            if 'train' in encoded_file:
                train_encode_list = np.load(os.path.join(encode_folder, encoded_file))
            elif 'test' in encoded_file:
                test_news_encode_list = np.load(os.path.join(encode_folder, encoded_file))
            elif 'news_t' in encoded_file:
                test_news_encode_list = np.load(os.path.join(encode_folder, encoded_file))
                train_encode_list = np.load(os.path.join(encode_folder, "encoded_selected_train_politifact.jsonl.npy"))
    train_news_encode_list = np.asarray([train_encode_list[i] for i in train_index_list])
    assert len(train_news_encode_list) == len(train_news_list)
    assert len(test_news_encode_list) == len(test_news_list)

    fake_num = 2
    real_num = 2

    debug = 1
    start2 = time.time()
    if debug == 0:
        dem_news_index_list = prompt_news_select(train_news_list, test_news_list, train_news_encode_list,
                                                 test_news_encode_list, fake_num, real_num)
    else:
        n_dem = len(test_news_list)
        sampled_list = list(range(len(train_news_list)))
        dem_news_index_list = [random.sample(sampled_list, 4) for _ in range(n_dem)]
    end2 = time.time()
    with open('time2.txt', 'w') as t:
        t.writelines('time_Inside_Investigation: {} seconds'.format(end2 - start2))

    dem_news_list = [[train_news_list[i] for i in index] for index in dem_news_index_list]

    with open(os.path.join(target_folder, 'dem_news_{}shot'.format(shot) + test_news_dataset), 'w') as f:
        json.dump(dem_news_list, f)

    assert len(dem_news_list) == len(test_news_list)

    start4 = time.time()
    LLM_save(shot, test_news_dataset, target_folder, test_news_list, google_use=False,
             dem_list=dem_news_list, output_folder=output_folder)
    end4 = time.time()
    with open('time4.txt', 'w') as t:
        t.writelines('time_Inside_Judge: {} seconds'.format(end4 - start4))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--shot', type=int, default=100, choices=[100, 64, 32, 16, 8])
    parser.add_argument('--dataset', default='politifact', choices=['news_t', 'politifact', 'gossipcop'])
    args = parser.parse_args()

    source_folder = '../dataset_full'
    encode_folder = '../dataset_full/encode'
    target_folder = '../dataset_full/judge_result'
    output_folder = './output_all/'
    test_news_dataset = args.dataset

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    extract_train_news_and_ask_LLM(args.shot, test_news_dataset, source_folder, encode_folder,
                                   target_folder, output_folder)
