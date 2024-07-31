import os
import json
import sys
import argparse
from tqdm import tqdm
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:32"
import torch
from transformers import pipeline, AutoTokenizer
import multiprocessing as mp

prompt_keyword = ("As a news keyword extractor, your task is to extract the six most important keywords"
                  " from a given news text. "
                  "The keywords should include when, where, who, what, how and why the news happened. "
                  "Please give me the six keywords only. "
                  "My first suggestion request is {} ")

def keywords_detect(pipe, news_dict):

    tokenizer = AutoTokenizer.from_pretrained("../zephyr-7b-beta")

    if news_dict['title'] != ' ' or news_dict['text'] != ' ':
        news_context = ' '.join([news_dict['title'], news_dict['text']])
    else:
        news_context = news_dict['tweet']
    news_tokens_ids = tokenizer(news_context)

    if len(news_tokens_ids['input_ids']) < 5:
        news_context = ' '.join([news_context, news_dict['tweet']])
    if len(news_tokens_ids['input_ids']) > 513:
        news_context = tokenizer.decode(news_tokens_ids['input_ids'][1:513])
    news_context = news_context.strip()
    prompt_content = prompt_keyword.format("\"" + news_context + "\"")
    messages = [
        {
            "role": "system",
            "content": "You are a news keywords extractor who gives keywords to help understanding news",
        },
        {"role": "user", "content": prompt_content},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    # print(outputs[0]["generated_text"])
    text = outputs[0]["generated_text"][outputs[0]["generated_text"].find('<|assistant|>'):]
    print(text)
    index = ['1. ', '2. ', '3. ', '4. ', '5. ', '6. ']
    wl = [text[text.find(index[i]) + 3: text.find(index[i + 1])] for i in range(5)]
    keywords_list = []
    for w in wl:
        kw = w
        if kw.find('(') != -1:
            kw = kw[:kw.find('(')]
        if kw.find(':') != -1:
            kw = kw[:kw.find(':')]
        kw = kw.strip()
        keywords_list.append(kw)
    print(keywords_list)
    print()
    return keywords_list


def keywords_save(batch):
    dataset_folder = '../extracted_dataset/'
    keywords_folder = '../dataset_with_keywords'
    if not os.path.exists(keywords_folder):
        os.mkdir(keywords_folder)
    dataset_list = [dataset for dataset in os.listdir(dataset_folder) if 'news_t' in dataset]
    pipe = pipeline("text-generation", model="../zephyr-7b-beta", torch_dtype=torch.bfloat16,
                    device_map="cuda")
    for dataset in dataset_list:
        #context = mp.get_context('spawn')
        #pool = context.Pool(2)
        print(dataset)
        with open(os.path.join(dataset_folder, dataset), 'r') as f:
            #news_list0 = f.readlines()[0]
            news_list0 = json.load(f)
        batch_size = int(len(news_list0))
        news_list = news_list0[:100]    #[batch*batch_size:(batch+1)*batch_size]
        print("len_news_list: {}".format(len(news_list)))
        # print(batch_size)
        keywords_list = []
        index_list = []
        for index, news in tqdm(enumerate(news_list), total=len(news_list)):
            print("No." + str(index))
            try:
                keywords_list.append(keywords_detect(pipe, news))
            except:
                index_list.append(index)

        if len(index_list) > 0:
            for index in index_list:
                print("No." + str(index))
                try:
                    keywords = keywords_detect(pipe, news_list[index])
                    keywords_list.insert(index, keywords)
                except:
                    text = news_list['title'] + news_list['text'] + news_list['tweet']
                    keywords_list.insert(index, text.strip().split(' ')[:5])

        for news_dict, keywords in tqdm(zip(news_list, keywords_list), total=len(news_list)):
            news_dict['keywords'] = keywords

        with open(os.path.join(keywords_folder, dataset), 'w') as fout:
            json.dump(news_list, fout)


if __name__ == "__main__":
    start1 = time.time()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--batch', type=int, default=0)
    args = parser.parse_args()
    keywords_save(args.batch)
    end1 = time.time()
    with open('time1.txt', 'w') as t:
        t.writelines('time_Extraction: {} seconds'.format(end1 - start1))