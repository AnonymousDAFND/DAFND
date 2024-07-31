import argparse
import os
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
import json
import torch
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm


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


prompt_determine = ("I need your assistance in evaluating the authenticity of a news article. "
                    "This news article include news title, news text and news tweet."
                    "The news article is: {}."
                    "There are two different views on this news article."
                    "Some people believe that {}, their explanation is: {}"
                    "Others believe that {}, their explanation is: {}"
                    "Please judge their opinion and give your decision."
                    "The first sentence after [Explanation] must be [This is fake news] or "
                    "[This is real news], and then give your explanation. "
                    "[Explanation]:"
                    )


def dict_append(dict_n, index, news, dev_decision, dev_reason, google_decision, google_reason,
                final_decision, final_reason):
    dict_n['index'].append(index)
    dict_n['label'].append(news['label'])
    dict_n['keywords'].append(news['keywords'])
    dict_n['title'].append(news['title'])
    dict_n['text'].append(news['text'])
    dict_n['tweet'].append(news['tweet'])
    dict_n['google'].append(news['google'])
    dict_n['dev_decision'].append(dev_decision)
    dict_n['dev_reason'].append(dev_reason)
    dict_n['google_decision'].append(google_decision)
    dict_n['google_reason'].append(google_reason)
    dict_n['final_decision'].append(final_decision)
    dict_n['final_reason'].append(final_reason)


def f1score(tp, fp, fn):
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r)
    return f1


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


def determine(dataset, shot, judge_result_folder, determine_result_folder, dataset_folder, output_folder):
    correct_detect = 0
    correct_fake = 0
    correct_real = 0
    wrong_fake = 0
    wrong_real = 0
    news_num = 0
    max_correct = 0
    dev_right_google_right = {'index': [], 'label': [], 'keywords': [], 'title': [],
                              'text': [], 'tweet': [], 'google': [],
                              'dev_decision': [], 'dev_reason': [],
                              'google_decision': [], 'google_reason': [],
                              'final_decision': [], 'final_reason': []}
    dev_wrong_google_right = {'index': [], 'label': [], 'keywords': [], 'title': [],
                              'text': [], 'tweet': [], 'google': [],
                              'dev_decision': [], 'dev_reason': [],
                              'google_decision': [], 'google_reason': [],
                              'final_decision': [], 'final_reason': []}
    dev_right_google_wrong = {'index': [], 'label': [], 'keywords': [],
                              'title': [], 'text': [], 'tweet': [], 'google': [],
                              'dev_decision': [], 'dev_reason': [],
                              'google_decision': [], 'google_reason': [],
                              'final_decision': [], 'final_reason': []}
    wrong_determine = {'index': [], 'label': [], 'keywords': [],
                       'title': [], 'text': [], 'tweet': [], 'google': [],
                       'dev_decision': [], 'dev_reason': [],
                       'google_decision': [], 'google_reason': [],
                       'final_decision': [], 'final_reason': []}

    result_jsonl_files = [file for file in os.listdir(judge_result_folder) if dataset in file]
    for result_jsonl in result_jsonl_files:
        if 'dem_result_no_google' in result_jsonl and str(shot) in result_jsonl:
            with open(os.path.join(judge_result_folder, result_jsonl), 'r') as f:
                dev_result_dict = json.load(f)
        elif 'google_result' in result_jsonl:
            with open(os.path.join(judge_result_folder, result_jsonl), 'r') as f:
                google_result_dict = json.load(f)
    test_jsonl_files = [file for file in os.listdir(dataset_folder) if dataset in file]
    for test_jsonl in test_jsonl_files:
        if 'test' in test_jsonl:
            with open(os.path.join(dataset_folder, test_jsonl), 'r') as f:
                test_news_list = json.load(f)
            break
        elif 'news_t' in test_jsonl:
            with open(os.path.join(dataset_folder, test_jsonl), 'r') as f:
                test_news_list = json.load(f)
            break
    assert len(test_news_list) == len(dev_result_dict['decision']) == len(google_result_dict['decision'])
    index_list = []
    pipe = pipeline("text-generation", model="../zephyr-7b-beta", torch_dtype=torch.bfloat16,
                    device_map="cuda")
    for index, (test_news, dev_decision, dev_reason, google_decision, google_reason) in tqdm(enumerate(
            zip(test_news_list, dev_result_dict['decision'], dev_result_dict['reason'],
                google_result_dict['decision'], google_result_dict['reason'])), total=len(test_news_list)):

        print(f'No.{index}')
        news_text, news_tweet = news_cut(test_news, n=160, m=160)
        news_article = ("news title: {}, news text: {}, news tweet: {}".format
                        (test_news['title'], news_text, news_tweet))
        if dev_decision == google_decision or (dev_decision in [
            'no_idea', 'no idea', ' '] and google_decision in ['no_idea', 'no idea', ' ']):
            if dev_decision == test_news['label']:
                correct_detect += 1
                max_correct += 1
                dict_append(dev_right_google_right, index, test_news, dev_decision, dev_reason,
                            google_decision, google_reason, test_news['label'], '')
                if test_news['label'] == 'fake':
                    correct_fake += 1
                else:
                    correct_real += 1
            else:
                dict_append(wrong_determine, index, test_news, dev_decision, dev_reason,
                            google_decision, google_reason, dev_decision, '')
                if dev_decision == 'fake':
                    wrong_fake += 1
                elif dev_decision == 'real':
                    wrong_real += 1
                elif dev_decision == 'no_idea' or dev_decision == ' ' or dev_decision == 'no idea':
                    if test_news['label'] == 'fake':
                        wrong_real += 1
                    else:
                        wrong_fake += 1
            news_num += 1
        else:
            if dev_decision == 'real':
                dd = 'this is real news'
            elif dev_decision == 'fake':
                dd = 'this is fake news'
            else:
                dd = 'no_idea'
            if google_decision == 'real':
                gd = 'this is real news'
            elif google_decision == 'fake':
                gd = 'this is fake news'
            else:
                gd = 'no_idea'
            tokenizer = AutoTokenizer.from_pretrained("../zephyr-7b-beta")
            thr_n = 80
            thr_m = 80
            dev_reason_ids = tokenizer(dev_reason)
            if len(dev_reason_ids['input_ids']) > thr_n:
                dr = tokenizer.decode(dev_reason_ids['input_ids'][1:thr_n])
            else:
                dr = dev_reason
            dr = dr.strip()

            google_reason_ids = tokenizer(google_reason)
            if len(google_reason_ids['input_ids']) > thr_m:
                gr = tokenizer.decode(google_reason_ids['input_ids'][1:thr_m])
            else:
                gr = google_reason
            gr = gr.strip()
            prompt_content = prompt_determine.format(news_article, dd, dr, gd, gr)
            try:
                decision, reason = LLM_detected(pipe, prompt_content)
                if decision == test_news['label']:
                    max_correct += 1
                    correct_detect += 1
                    if test_news['label'] == dev_decision:
                        dict_append(dev_right_google_wrong, index, test_news, dev_decision, dev_reason,
                                    google_decision, google_reason, test_news['label'], reason)
                    elif test_news['label'] == google_decision:
                        dict_append(dev_wrong_google_right, index, test_news, dev_decision, dev_reason,
                                    google_decision, google_reason, test_news['label'], reason)
                    if test_news['label'] == 'fake':
                        correct_fake += 1
                    else:
                        correct_real += 1
                else:
                    dict_append(wrong_determine, index, test_news, dev_decision, dev_reason,
                                google_decision, google_reason, decision, reason)
                    if test_news['label'] == dev_decision or test_news['label'] == google_decision:
                        max_correct += 1
                    if decision == 'fake':
                        wrong_fake += 1
                    elif decision == 'real':
                        wrong_real += 1
                    elif decision == 'no_idea' or decision == ' ' or decision == 'no idea':
                        if test_news['label'] == 'fake':
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
            dev_decision = dev_result_dict['decision'][index]
            dev_reason = dev_result_dict['reason'][index]
            google_decision = google_result_dict['decision'][index]
            google_reason = google_result_dict['reason'][index]

            if dev_decision == 'real':
                dd = 'this is real news'
            elif dev_decision == 'fake':
                dd = 'this is fake news'
            else:
                dd = 'no_idea'
            if google_decision == 'real':
                gd = 'this is real news'
            elif google_decision == 'fake':
                gd = 'this is fake news'
            else:
                gd = 'no_idea'

            news_article = ("news title: {}, news text: {}, news tweet: {}".format
                            (test_news['title'], test_news['text'], test_news['tweet']))
            tokenizer = AutoTokenizer.from_pretrained("../zephyr-7b-beta")
            thr_n = 80
            thr_m = 80
            dev_reason_ids = tokenizer(dev_reason)
            if len(dev_reason_ids['input_ids']) > thr_n:
                dr = tokenizer.decode(dev_reason_ids['input_ids'][1:thr_n])
            else:
                dr = dev_reason
            dr = dr.strip()

            google_reason_ids = tokenizer(google_reason)
            if len(google_reason_ids['input_ids']) > thr_m:
                gr = tokenizer.decode(google_reason_ids['input_ids'][1:thr_m])
            else:
                gr = google_reason
            gr = gr.strip()
            prompt_content = prompt_determine.format(news_article, dd, dr, gd, gr)
            try:
                decision, reason = LLM_detected(pipe, prompt_content)
                if decision == test_news['label']:
                    max_correct += 1
                    correct_detect += 1
                    if test_news['label'] == dev_decision:
                        dict_append(dev_right_google_wrong, index, test_news, dev_decision, dev_reason,
                                    google_decision, google_reason, test_news['label'], reason)
                    elif test_news['label'] == google_decision:
                        dict_append(dev_wrong_google_right, index, test_news, dev_decision, dev_reason,
                                    google_decision, google_reason, test_news['label'], reason)
                    if test_news['label'] == 'fake':
                        correct_fake += 1
                    else:
                        correct_real += 1
                else:
                    dict_append(wrong_determine, index, test_news, dev_decision, dev_reason,
                                google_decision, google_reason, decision, reason)
                    if test_news['label'] == dev_decision or test_news['label'] == google_decision:
                        max_correct += 1
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

    acc = correct_detect / news_num
    max_acc = max_correct / news_num
    f1 = f1score(correct_fake, wrong_fake, wrong_real)
    output_file = 'determine_{}_{}shot_{}.txt'.format(dataset, shot,
                                                      time.strftime("%Y%m%d%H%M%S", time.localtime()))
    with open(os.path.join(output_folder, output_file), 'w') as f:
        result = [
            "{0} Accuracy: {1:.4f} ({2}/{3})  Max_Accuracy: {4:.4f} ({5}/{6})\n".format(dataset, acc, correct_detect,
                                                                                        news_num, max_acc, max_correct,
                                                                                        news_num),
            "true_fake: {} true_real: {} false_fake: {} false_real: {}\n".format(correct_fake, correct_real,
                                                                                       wrong_fake, wrong_real),
            "F1 score: {0:.4f}".format(f1)]
        f.writelines(result)
    with open(os.path.join(determine_result_folder, 'dev_right_google_right_{}shot_'.format(shot) + dataset), 'w') as f:
        json.dump(dev_right_google_right, f)
    with open(os.path.join(determine_result_folder, 'dev_right_google_wrong_{}shot_'.format(shot) + dataset), 'w') as f:
        json.dump(dev_right_google_wrong, f)
    with open(os.path.join(determine_result_folder, 'dev_wrong_google_right_{}shot_'.format(shot) + dataset), 'w') as f:
        json.dump(dev_wrong_google_right, f)
    with open(os.path.join(determine_result_folder, 'wrong_determine_{}shot_'.format(shot) + dataset), 'w') as f:
        json.dump(wrong_determine, f)

if __name__ == '__main__':
    start6 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--shot', type=int, default=100, choices=[100, 64, 32, 16, 8])
    parser.add_argument('--dataset', default='news_t', choices=['news_t', 'politifact', 'gossipcop'])
    args = parser.parse_args()

    dataset_folder = '../dataset_full'
    judge_result_folder = '../dataset_full/judge_result'
    determine_result_folder = '../dataset_full/determine_result'
    output_folder = './output_all/'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists(determine_result_folder):
        os.mkdir(determine_result_folder)
    determine(args.dataset, args.shot, judge_result_folder, determine_result_folder,
              dataset_folder, output_folder=output_folder)

    end6 = time.time()
    with open('time6.txt', 'w') as t:
        t.writelines('time_Determine: {} seconds'.format(end6 - start6))