import argparse
import os
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
import json
import torch
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm


def f1score(tp, fp, fn):
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r)
    return f1


def LLM_detected(pipe, prompt_content):

    messages = [
        {
            "role": "system",
            "content": "You are an expert of news authenticity evaluation.",
        },
        {"role": "user", "content": prompt_content},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    print(outputs[0]["generated_text"])
    a_text = outputs[0]["generated_text"][outputs[0]["generated_text"].find('<|assistant|>'):]
    if "[This is fake news]" in a_text[:40]:
        prediction = 'fake'
    elif "[This is real news]" in a_text[:40]:
        prediction = 'real'
    elif a_text.count("[This is fake news]") > a_text.count("[This is real news]"):
        prediction = 'fake'
    elif a_text.count("[This is fake news]") < a_text.count("[This is real news]"):
        prediction = 'real'
    elif a_text.count("is fake") > a_text.count("is real"):
        prediction = 'fake'
    elif a_text.count("is fake") < a_text.count("is real"):
        prediction = 'real'
    elif a_text.count("is not real") > a_text.count("is not fake"):
        prediction = 'fake'
    elif a_text.count("is not real") < a_text.count("is not fake"):
        prediction = 'real'
    elif a_text.count("fake") > a_text.count("real"):
        prediction = 'fake'
    elif a_text.count("fake") < a_text.count("real"):
        prediction = 'real'
    else:
        prediction = 'no_idea'
    return prediction, a_text


prompt_google_search = ("I need your assistance in evaluating the authenticity of a news article. "
                        "I will provide you the news article and additional information about this news. "
                        "Please analyze the following news and give your decision. "
                        "The first sentence of your [Decision] must be [This is fake news] or [This is real news]. "
                        "The news article is: {}"
                        "The additional information is: {}"
                        "[Decision]:"
                        )


def news_cut(news, n, m):
    tokenizer = AutoTokenizer.from_pretrained("../zephyr-7b-beta")
    thr_n = n+1
    thr_m = m+1

    news_text_ids = tokenizer(news['text'])
    if len(news_text_ids['input_ids']) > thr_n:
        news_text = tokenizer.decode(news_text_ids['input_ids'][1:thr_n])
    else:
        news_text = news['text']
    news_text = news_text.strip()

    news_tweet_ids = tokenizer(news['tweet'])
    if len(news_tweet_ids['input_ids']) > thr_m:
        news_tweet = tokenizer.decode(news_tweet_ids['input_ids'][1:thr_m])
    else:
        news_tweet = news['tweet']
    news_tweet = news_tweet.strip()
    return news_text, news_tweet

def google_inv(test_news_dataset, source_folder, target_folder, output_folder):
    correct_detect = 0
    correct_fake = 0;correct_real = 0;wrong_fake = 0;wrong_real = 0
    news_num = 0

    news_jsonl_files = [file for file in os.listdir(source_folder) if test_news_dataset in file]
    for news_jsonl in news_jsonl_files:
        if 'test' in news_jsonl:
            with open(os.path.join(source_folder, news_jsonl), 'r') as f:
                test_news_list = json.load(f)
        elif 'train' in news_jsonl:
            with open(os.path.join(source_folder, news_jsonl), 'r') as f:
                train_news_list = json.load(f)
        elif 'news_t' in news_jsonl:
            with open(os.path.join(source_folder, news_jsonl), 'r') as f:
                test_news_list = json.load(f)
            with open(os.path.join(source_folder, "selected_train_politifact.jsonl"), 'r') as f:
                train_news_list = json.load(f)

    index_list = []
    decision_list = []
    reason_list = []

    pipe = pipeline("text-generation", model="../zephyr-7b-beta", torch_dtype=torch.bfloat16,
                    device_map="cuda")
    for index, news in tqdm(enumerate(test_news_list), total=len(test_news_list)):
        print(f'No.{index}')
        news_text, news_tweet = news_cut(news, n=160, m=160)
        news_article = ("news title: {}, news text: {}, news tweet: {}".format
                        (news['title'], news_text, news_tweet))
        prompt_content = prompt_google_search.format(news_article, news['google'])
        try:
            decision, reason = LLM_detected(pipe, prompt_content)
            decision_list.append(decision)
            reason_list.append(reason)
            if decision == news['label']:
                correct_detect += 1
                if news['label'] == 'fake':
                    correct_fake += 1
                else:
                    correct_real += 1
            else:
                if decision == 'fake':
                    wrong_fake += 1
                elif decision == 'real':
                    wrong_real += 1
                elif decision == 'no_idea':
                    if news['label'] == 'fake':
                        wrong_real += 1
                    else:
                        wrong_fake += 1
            news_num += 1
        except:
            print("can't load this news!!! len:{}".format(len(prompt_content)))
            print("Unexpected error:", sys.exc_info()[0])
            index_list.append(index)
            time.sleep(30)

    if len(index_list) > 0:
        for index in tqdm(index_list, total=len(index_list)):
            print(f'No.{index}')
            test_news = test_news_list[index]
            news_article = ("news title: {}, news text: {}, news tweet: {}".format
                            (test_news['title'], test_news['text'], test_news['tweet']))
            prompt_content = prompt_google_search.format(news_article, test_news['google'])
            try:
                decision, reason = LLM_detected(pipe, prompt_content)
                decision_list.insert(index, decision)
                reason_list.insert(index, reason)
                if decision == test_news['label']:
                    correct_detect += 1
                    if test_news['label'] == 'fake':
                        correct_fake += 1
                    else:
                        correct_real += 1
                else:
                    if decision == 'fake':
                        wrong_fake += 1
                    elif decision == 'real':
                        wrong_real += 1
                    elif decision == 'no_idea':
                        if test_news['label'] == 'fake':
                            wrong_real += 1
                        else:
                            wrong_fake += 1
                news_num += 1
            except:
                print("can't load this news!!! len:{}".format(len(prompt_content)))
                print("Unexpected error:", sys.exc_info()[0])
                decision_list.insert(index, 'no_idea')
                reason_list.insert(index, 'no_idea')

    result = {'decision': decision_list, 'reason': reason_list}
    with open(os.path.join(target_folder, 'google_result_' + test_news_dataset), 'w') as f:
        json.dump(result, f)
    acc = correct_detect / news_num
    f1 = f1score(correct_fake, wrong_fake, wrong_real)
    output_file = 'google_{}_{}.txt'.format(test_news_dataset, time.strftime("%Y%m%d%H%M%S", time.localtime()))
    with open(os.path.join(output_folder, output_file), 'w') as f:
        result = ["{0} Accuracy: {1:.4f} ({2}/{3})\n".format(test_news_dataset, acc, correct_detect, news_num),
                  "correct_fake: {} correct_real: {} wrong_fake: {} wrong_real: {}\n".format(correct_fake, correct_real,
                                                                                             wrong_fake, wrong_real),
                  "F1 score: {0:.4f}".format(f1)]
        f.writelines(result)


if __name__ == "__main__":
    start5 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='news_t', choices=['news_t', 'politifact', 'gossipcop'])
    args = parser.parse_args()
    test_news_dataset = args.dataset
    source_folder = '../dataset_full'
    target_folder = '../dataset_full/judge_result'
    output_folder = './output_all/'
    print(test_news_dataset)
    google_inv(test_news_dataset, source_folder, target_folder, output_folder)
    end5 = time.time()
    with open('time5.txt', 'w') as t:
        t.writelines('time_Outside_Judge: {} seconds'.format(end5 - start5))