import os
import json
import random
from serpapi import GoogleSearch
import time
from IPython.utils import io
from tqdm import tqdm
import argparse


def google_search(news_keywords):
    params = {
        "engine": "google",
        "q": news_keywords,
        "api_key":  ###
    }
    with io.capture_output() as captured:
        search = GoogleSearch(params)
        res = search.get_dict()
    answer = None
    snippet = None
    title = None

    if "answer_box" in res.keys() and "answer" in res["answer_box"].keys():
        answer = res["answer_box"]["answer"]
    if "answer_box" in res.keys() and "snippet" in res["answer_box"].keys():
        snippet = res["answer_box"]["snippet"]
        title = res["answer_box"]["title"]

    elif (
            "answer_box" in res.keys()
            and "contents" in res["answer_box"].keys()
            and "table" in res["answer_box"]["contents"].keys()
    ):
        snippet = res["answer_box"]["contents"]["table"]
        title = res["answer_box"]["title"]
    elif "answer_box" in res.keys() and "list" in res["answer_box"].keys():
        snippet = res["answer_box"]["list"]
        title = res["answer_box"]["title"]
    elif "organic_results" in res:
        for i in range(len(res["organic_results"])):
            if "snippet" in res["organic_results"][i].keys() and "title" in res["organic_results"][i].keys() and len(res["organic_results"][i]["snippet"]):
                snippet = res["organic_results"][i]["snippet"]
                title = res["organic_results"][i]["title"]
                break
    elif (
            "organic_results" in res
            and "rich_snippet" in res["organic_results"][0].keys()
    ):
        snippet = res["organic_results"][0]["rich_snippet"]
        title = res["organic_results"][0]["title"]
    else:
        snippet = None
    if snippet is not None:
        # title = title.replace("- Wikipedia", "").strip()
        toret = f"{title}: {snippet}"
        toret = f"{toret} So the answer is {answer}." if answer is not None else toret
    else:
        toret = ""
    return toret, res




def fix():
    path = '../dataset_full/google_search/google_selected_valid_politifact.jsonl'
    with open(path, 'r') as f:
        a = json.load(f)
    l = a.keys()
    d = {}
    for i in l:
        n = i[:-4]
        s = i[-4:]
        d[n+'_'+s] = a[i]
    with open(path, 'w') as f:
        json.dump(d, f)


def google_save(datafile, a, b):
    keywords_folder = '../dataset_with_keywords'
    selected_folder = '../dataset_full/selected_file'
    full_folder = '../dataset_full'
    google_folder = '../dataset_full/google_search'
    if not os.path.exists(full_folder):
        os.mkdir(full_folder)
    if not os.path.exists(google_folder):
        os.mkdir(google_folder)
    folder = selected_folder if 'selected' in datafile else keywords_folder
    with open(os.path.join(folder, datafile), 'r') as f:
        #news_list = f.readlines()[0]
        news_list = json.load(f)
    news_list = news_list[a: b]
    keywords_list = [' '.join(news['keywords']) for news in news_list]
    search_res_list = []
    search_toret_list = []
    search_result = {}
    for keywords in tqdm(keywords_list, total=len(news_list)):
        toret, res = google_search(keywords)
        search_res_list.append(res)
        search_toret_list.append(toret)
    #time.sleep(130)
    # search_result_list = list(tqdm(map(google_search, keywords_list), total=len(news_list)))
    assert len(news_list) == len(search_res_list) == len(search_toret_list)
    for news_dict, toret, res in tqdm(zip(news_list, search_toret_list, search_res_list), total=len(news_list)):
        news_dict['google'] = toret
        search_result[news_dict['news_id']+'_'+news_dict['label']] = res
    output_name = datafile
    with open(os.path.join(full_folder, output_name), 'w') as fout:
        json.dump(news_list, fout)
    with open(os.path.join(google_folder, 'google_' + output_name), 'w') as fout:
        json.dump(search_result, fout)


if __name__ == '__main__':
    start3 = time.time()

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--datafile',
                        default='news_t.jsonl',
                        choices=['selected_valid_politifact.jsonl', 'selected_train_politifact.jsonl',
                         'politifact_test.jsonl', 'selected_valid_gossipcop.jsonl', 'selected_train_gossipcop.jsonl',
                          'gossipcop_test.jsonl', 'news_t.jsonl'])
    parser.add_argument('--a', type=int, default=0)
    parser.add_argument('--b', type=int, default=100)
    args = parser.parse_args()
    print(f"{args.datafile}[{args.a}: {args.b}]")
    google_save(args.datafile, args.a, args.b)

    end3 = time.time()
    with open('time3.txt', 'w') as t:
        t.writelines('time_Outside_Investigation: {} seconds'.format(end3 - start3))
