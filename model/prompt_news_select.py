import json
import faiss
import numpy as np
from tqdm import tqdm
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def vec_compare(d, k, train_news_encode_list, news_content_encode):
    index = faiss.IndexFlatL2(d)
    index.add(train_news_encode_list)
    D, I = index.search(news_content_encode, k)
    return I


def keywords_compare(train_news_list, test_news_list):
    train_news_index = []
    for test_news in test_news_list:
        same_num = []
        keywords_set = set(' '.join(test_news['keywords']).lower().split(' '))
        for index, train_news in enumerate(train_news_list):
            train_keywords_set = set(' '.join(train_news['keywords']).lower().split(' '))
            kw_intersection = keywords_set.intersection(train_keywords_set)
            same_num.append((index, len(kw_intersection)))
        same_num.sort(key=lambda a: a[1], reverse=True)
        train_news_index.append([i[0] for i in same_num[:1]])
    return train_news_index


def prompt_news_select(train_news_list, test_news_list, train_news_encode_list, test_news_encode_list, fake_num=2,
                       real_num=2):

    fake_index_list = []
    fake_news_list = []
    real_index_list = []
    real_news_list = []

    for index, train_news in tqdm(enumerate(train_news_list), total=len(train_news_list)):
        if train_news['label'] == 'fake':
            fake_index_list.append(index)
            fake_news_list.append(train_news)
        else:
            real_index_list.append(index)
            real_news_list.append(train_news)

    d = len(train_news_encode_list[0])
    fake_vec_index_list = vec_compare(d, fake_num, train_news_encode_list[fake_index_list], test_news_encode_list)
    real_vec_index_list = vec_compare(d, real_num, train_news_encode_list[real_index_list], test_news_encode_list)
    #vec_index_list = vec_compare(d, 1, train_news_encode_list, test_news_encode_list)

    #fake_kw_index_list = keywords_compare(fake_news_list, test_news_list)
    #real_kw_index_list = keywords_compare(real_news_list, test_news_list)
    #kw_index_list = keywords_compare(train_news_list, test_news_list)

    fake_vec_prompt_index_list = np.asarray([[fake_index_list[j] for j in i] for i in fake_vec_index_list])
    real_vec_prompt_index_list = np.asarray([[real_index_list[j] for j in i] for i in real_vec_index_list])
    #fake_kw_prompt_index_list = np.asarray([[fake_index_list[j] for j in i] for i in fake_kw_index_list])
    #real_kw_prompt_index_list = np.asarray([[real_index_list[j] for j in i] for i in real_kw_index_list])

    prompt_index_list = np.concatenate((fake_vec_prompt_index_list, real_vec_prompt_index_list,), axis=1)
    return prompt_index_list


def test(train_news_dataset, train_news_encode, test_news_encode):
    test_news_dataset = '../dataset_with_keywords/politifact_test.jsonl'
    with open(test_news_dataset, 'r') as f:
        test_news_list = json.load(f)
    test_news_index_list = [33, ]  # test_news_list[33]
    test_list = [test_news_list[33]]
    for test_news_index in test_news_index_list:
        print(test_news_list[test_news_index])
    with open(train_news_dataset, 'r') as f:
        train_news_list = json.load(f)
    train_news_encode_list = np.load(train_news_encode)
    test_news_encode_list = np.load(test_news_encode)

    index_list = prompt_news_select(train_news_list, test_news_list, train_news_encode_list,
                                    test_news_encode_list)
    for index in index_list:
        print(train_news_list[index])


if __name__ == '__main__':
    train_news_dataset = '../dataset_with_keywords/politifact_train.jsonl'
    train_news_encode = '../dataset_with_keywords/encoded_keywords_dataset/encoded_politifact_train.jsonl.npy'
    test_news_encode = '../dataset_with_keywords/encoded_keywords_dataset/encoded_politifact_test.jsonl.npy'
    test(train_news_dataset, train_news_encode, test_news_encode)
