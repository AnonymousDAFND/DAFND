import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,0,2'
import json
from LLM_detected_and_save import LLM_save
from prompt_news_select import prompt_news_select
import numpy as np


def dem_and_google(test_news_dataset, source_folder, encode_folder, target_folder, output_folder):
    news_jsonl_files = [file for file in os.listdir(source_folder) if test_news_dataset in file]
    for news_jsonl in news_jsonl_files:
        if 'test' in news_jsonl:
            with open(os.path.join(source_folder, news_jsonl), 'r') as f:
                test_news_list = json.load(f)
        elif 'train' in news_jsonl:
            with open(os.path.join(source_folder, news_jsonl), 'r') as f:
                train_news_list = json.load(f)
    train_news_encode_list = []
    test_news_encode_list = []
    for encoded_file in os.listdir(encode_folder):
        if test_news_dataset in encoded_file:
            if 'train' in encoded_file:
                train_news_encode_list = np.load(os.path.join(encode_folder, encoded_file))
            elif 'test' in encoded_file:
                test_news_encode_list = np.load(os.path.join(encode_folder, encoded_file))
    assert len(train_news_encode_list) == len(train_news_list)

    fake_num = 2
    real_num = 2

    dem_news_index_list = prompt_news_select(train_news_list, test_news_list, train_news_encode_list,
                                             test_news_encode_list, fake_num, real_num)
    dem_news_list = [[train_news_list[i] for i in index] for index in dem_news_index_list]
    assert len(dem_news_list) == len(test_news_list)

    LLM_save(int(len(train_news_list)/2), test_news_dataset, target_folder, test_news_list, google_use=True,
             dem_list=dem_news_list, output_folder=output_folder)


if __name__ == '__main__':
    source_folder = '../dataset_full'
    encode_folder = '../dataset_full/encode'
    target_folder = '../dataset_full/judge_result'
    output_folder = './output_all/'
    test_news_dataset = 'politifact'

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    dem_and_google(test_news_dataset, source_folder, encode_folder,
                   target_folder, output_folder)
