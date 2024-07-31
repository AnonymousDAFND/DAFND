import os
import json
import random
from serpapi import GoogleSearch
import torch
from IPython.utils import io
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer
import argparse

stop_title = ['YouTube', 'dscc', 'dailynative.us', 'On and On',
              'CQ.com', 'Outlook, Office, Skype, Bing, Breaking News, and Latest Videos', 'Account Suspended',
              'Resource Not Available']
stop_text = [
    "Oops! It Looks like the page you are looking for has been removed.  We're redirecting you to our homepage...",
    'The page you requested cannot be found at this time. It may be temporarily unavailable or it may have been removed or relocated. '
    ' See one of the following pages for possible options:',
    'Tweet with a location  You can add location information to your Tweets, such as your city or precise location, from the web and via third-party applications. You always have the option to delete your Tweet location history. Learn more',
    "The page you're looking for isn't here.  Either someone gave you a bad link or there's something funky going on. Either way, we're truly sorry for the inconvenience.",
    'The interactive transcript could not be loaded.  Rating is available when the video has been rented.  This feature is not available right now. Please try again later.']


def fix():
    full_folder = '../dataset_full'
    selected_folder = '../dataset_full/selected_file'
    with open(os.path.join(full_folder, 'selected_valid_politifact.jsonl'), 'r') as f1:
        g_list = f1.readlines()[0]
        g_list = json.loads(g_list)
    with open(os.path.join(selected_folder, 'selected_valid_politifact.jsonl'), 'r') as f2:
        k_list = f2.readlines()[0]
        k_list = json.loads(k_list)
    for i in range(len(k_list)):
        k_list[i]['google'] = g_list[i]['google']
    with open(os.path.join(full_folder, 'selected_valid_politifact.jsonl'), 'w') as f:
        json.dump(k_list, f)


def train_data_select():
    keywords_folder = '../dataset_with_keywords'
    full_folder = '../dataset_full'
    selected_folder = '../dataset_full/selected_file'
    if not os.path.exists(full_folder):
        os.mkdir(full_folder)
    if not os.path.exists(selected_folder):
        os.mkdir(selected_folder)
    train_dataset_list = [ds for ds in os.listdir(keywords_folder) if 'gossipcop_train' in ds]
    for dataset in train_dataset_list:
        #tokenizer = AutoTokenizer.from_pretrained("../zephyr-7b-beta")
        with open(os.path.join(keywords_folder, dataset), 'r') as f:
            news_list = f.readlines()[0]
            news_list = json.loads(news_list)
        good_fake_news_list = []
        good_real_news_list = []
        for news in news_list:
            if news['title'] in stop_title or 'on Twitter' in news['title'] or news['text'] in stop_text or 'Trendolizer™' in news['text'] or len(news['text']) < 2:
                continue
            if news['label'] == 'fake':
                good_fake_news_list.append(news)
            elif news['label'] == 'real':
                good_real_news_list.append(news)
        print("fake_num:{}, real_num:{}".format(len(good_fake_news_list), len(good_real_news_list)))
        fake_gap_num = 200 - len(good_fake_news_list)
        real_gap_num = 200 - len(good_real_news_list)
        if fake_gap_num > 0:
            good_fake_news_list += [i for i in news_list if len(i['text']) == 1 and i['label'] == 'fake'][:fake_gap_num]
        if real_gap_num > 0:
            good_real_news_list += [i for i in news_list if len(i['text']) == 1 and i['label'] == 'real'][:real_gap_num]
        checked_fake_news_list = random.sample(good_fake_news_list, 200)
        checked_real_news_list = random.sample(good_real_news_list, 200)
        train_list = checked_fake_news_list[:100] + checked_real_news_list[:100]
        valid_list = checked_fake_news_list[100:] + checked_real_news_list[100:]
        random.shuffle(train_list)
        random.shuffle(valid_list)
        dataset_name = 'politifact.jsonl' if 'politifact' in dataset else 'gossipcop.jsonl'
        with open(os.path.join(selected_folder, "selected_train_" + dataset_name), 'w') as f:
            json.dump(train_list, f)
        with open(os.path.join(selected_folder, "selected_valid_" + dataset_name), 'w') as f:
            json.dump(valid_list, f)

if __name__ == '__main__':
    '''parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--datafile',
                        default='gossipcop_test.jsonl',
                        choices=['selected_valid_politifact.jsonl', 'selected_train_politifact.jsonl',
                         'politifact_test.jsonl', 'selected_valid_gossipcop.jsonl', 'selected_train_gossipcop.jsonl',
                          'gossipcop_test.jsonl'])
    args = parser.parse_args()
    google_save(args.datafile)'''
    #fix()
    train_data_select()