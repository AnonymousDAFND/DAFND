import os
import json
import time
import torch
from transformers import pipeline
import sys
from prompt import prompt_generate
from tqdm import tqdm


def LLM_detect(pipe, prompt_content, google_use=False):
    messages = [
        {
            "role": "system",
            "content": "You are an expert of news authenticity evaluation. As an expert of "
                       "news authenticity evaluation, you should analyze and evaluate"
                       "the authenticity of news",
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


def f1score(tp, fp, fn):
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r)
    return f1

def LLM_save(shot, test_news_dataset, target_folder, news_list, dem_list=None, google_use=False, output_folder='./output'):
    correct_detect = 0
    correct_fake = 0
    correct_real = 0
    wrong_fake = 0
    wrong_real = 0
    news_num = 0

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    print("total num: {}".format(len(news_list)))
    index_list = []
    decision_list = []
    reason_list = []
    pipe = pipeline("text-generation", model="../zephyr-7b-beta", torch_dtype=torch.bfloat16,
                    device_map="cuda")
    for index, (news, dem) in tqdm(enumerate(zip(news_list, dem_list)), total=len(news_list)):
        print(f'No.{index}')
        prompt_news = prompt_generate(news, dem, google_use)
        try:
            decision, reason = LLM_detect(pipe, prompt_news)
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
            print("can't load this news!!! ")
            print("Unexpected error:", sys.exc_info()[0])
            index_list.append(index)
            time.sleep(30)

    if len(index_list) > 0:
        for index in tqdm(index_list, total=len(index_list)):
            '''pipe = pipeline("text-generation", model="../zephyr-7b-beta", torch_dtype=torch.bfloat16,
                            device_map="auto")'''
            print(f'No.{index}')
            news = news_list[index]
            dem = dem_list[index]
            prompt_content = prompt_generate(news, dem)
            try:
                decision, reason = LLM_detect(pipe, prompt_content)
                decision_list.insert(index, decision)
                reason_list.insert(index, reason)
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
                print("can't load this news!!!")
                print("Unexpected error:", sys.exc_info()[0])
                decision_list.insert(index, 'no_idea')
                reason_list.insert(index, 'no_idea')

    result = {'decision': decision_list, 'reason': reason_list}

    if google_use:
        output_file = 'dem_{}_{}_{}shot_{}.txt'.format(test_news_dataset, 'use_google', shot,
                                                       time.strftime("%Y%m%d%H%M%S", time.localtime()))
        result_file = 'dem_result_use_google_{}_{}shot'.format(test_news_dataset, shot)
    else:
        output_file = 'dem_{}_{}_{}shot_{}.txt'.format(test_news_dataset, 'no_google', shot,
                                                       time.strftime("%Y%m%d%H%M%S", time.localtime()))
        result_file = 'dem_result_no_google_{}_{}shot'.format(test_news_dataset, shot)
    with open(os.path.join(target_folder, result_file), 'w') as f:
        json.dump(result, f)

    acc = correct_detect / news_num
    f1 = f1score(correct_fake, wrong_fake, wrong_real)
    print("****************************  RESULT  ********************************")
    print("")
    print("{0} Accuracy: {1:.4f} ({2}/{3})".format(news_list[0]['news_source'], acc, correct_detect, news_num))
    print(
        "correct_fake: {} correct_real: {} wrong_fake: {} wrong_real: {}".format(correct_fake, correct_real, wrong_fake,
                                                                                 wrong_real))
    print("F1 score: {0:.4f}".format(f1))
    print("**********************************************************************")
    with open(os.path.join(output_folder, output_file), 'w') as f:
        result = ["{0} Accuracy: {1:.4f} ({2}/{3})\n".format(test_news_dataset, acc,
                                                             correct_detect, news_num),
                  "correct_fake: {} correct_real: {} wrong_fake: {} wrong_real: {}\n".format(
                      correct_fake, correct_real, wrong_fake, wrong_real),
                  "F1 score: {0:.4f}".format(f1)
                  ]
        f.writelines(result)
