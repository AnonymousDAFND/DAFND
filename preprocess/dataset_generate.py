import json
import shutil
import os
import random

stop_title = ['YouTube', 'dscc', 'dailynative.us', 'On and On',
              'CQ.com', 'Outlook, Office, Skype, Bing, Breaking News, and Latest Videos', 'Account Suspended',
              'Resource Not Available']
stop_text = ["Oops! It Looks like the page you are looking for has been removed.  We're redirecting you to our homepage...",
             'The page you requested cannot be found at this time. It may be temporarily unavailable or it may have been removed or relocated. '
             ' See one of the following pages for possible options:',
             'Tweet with a location  You can add location information to your Tweets, such as your city or precise location, from the web and via third-party applications. You always have the option to delete your Tweet location history. Learn more',
             "The page you're looking for isn't here.  Either someone gave you a bad link or there's something funky going on. Either way, we're truly sorry for the inconvenience.",
             'The interactive transcript could not be loaded.  Rating is available when the video has been rented.  This feature is not available right now. Please try again later.']
def dataset_generate(sourcefolder, targetfolder, sourcedataset_list, extract_num):
    for dataset in sourcedataset_list:
        dataset_real_fake = []
        dataset_real_fake.append(dataset + '_fake')
        dataset_real_fake.append(dataset + '_real')
        test_news_jsonl = []
        train_news_jsonl = []
        '''targetdataset_path = os.path.join(targetfolder, dataset)
        if not os.path.exists(targetdataset_path):
            os.mkdir(targetdataset_path)'''
        for sourcedataset in dataset_real_fake:
            _, label = sourcedataset.split('_')
            sourcepath = os.path.join(sourcefolder, sourcedataset)
            all_news_list = []
            sourcenews_list = os.listdir(sourcepath)
            for sourcenews in sourcenews_list:
                sourcenews_path = os.path.join(sourcepath, sourcenews)
                with open(os.path.join(sourcenews_path, 'news_article.json'), 'r') as a, \
                     open(os.path.join(sourcenews_path, 'tweets.json'), 'r') as t:
                    news_article = json.load(a)
                    tweets = json.load(t)
                if len(news_article) == 0 or news_article['title'] in stop_title or 'on Twitter' in news_article['title'] or news_article['text'] in stop_text or 'Trendolizer™' in news_article['text'] or len(
                        news_article['text']) < 200 or len(news_article['text']) > 7500:
                    continue
                all_news_list.append(sourcenews)
            test_list = random.sample(all_news_list, int(extract_num[sourcedataset] * 0.2))
            if len(all_news_list) >= extract_num[sourcedataset]:
                train_list = random.sample(all_news_list, int(extract_num[sourcedataset] * 0.8))
            else:
                train_list = [news for news in all_news_list if news not in test_list]
                other_list = []
                for news in sourcenews_list:
                    if news not in train_list and news not in test_list:
                        other_list.append(news)
                train_list += random.sample(other_list, int(extract_num[sourcedataset] * 0.8) - len(train_list))
            news_dict_jsonl(test_list, sourcepath, label, test_news_jsonl)
            news_dict_jsonl(train_list, sourcepath, label, train_news_jsonl)
        random.shuffle(test_news_jsonl)
        random.shuffle(train_news_jsonl)
        save_jsonl(test_news_jsonl, targetfolder, dataset + '_test.jsonl')
        save_jsonl(train_news_jsonl, targetfolder, dataset + '_train.jsonl')

def save_jsonl(jsonl, path, filename):
    with open(os.path.join(path, filename), 'w') as f:
        json.dump(jsonl, f)
def news_dict_jsonl(news_list, sourcepath, label, json_list):
    for news in news_list:
        sourcenews_path = os.path.join(sourcepath, news)
        with open(os.path.join(sourcenews_path, 'news_article.json'), 'r') as a, \
                open(os.path.join(sourcenews_path, 'tweets.json'), 'r') as t:
            news_article = json.load(a)
            tweets = json.load(t)
        news_dict = {}
        news_id = news[10:]
        all_tweets_list = [tweet['text'] for tweet in tweets['tweets'] if len(tweet['text']) > 0]
        if len(all_tweets_list) > 20:
            tweets_list = random.sample(all_tweets_list, 20)
        else:
            tweets_list = all_tweets_list[:]
        news_dict['label'] = label
        news_dict['news_id'] = news_id
        news_dict['news_source'] = news_article['news_source'] if len(news_article) > 0 else ' '
        news_dict['title'] = news_article['title'] if len(news_article) > 0 else ' '
        news_dict['text'] = ' '.join(news_article['text'].split('\n')) if len(news_article) > 0 else ' '
        news_dict['tweet'] = ' '.join(' '.join(tweets_list).split('\n'))
        json_list.append(news_dict)



if __name__ == '__main__':
    sourcefolder = '../../FakeNewsNet_Dataset/'
    targetfolder = '../extracted_dataset/'
    sourcedataset_list = ['politifact', 'gossipcop']
    extract_num = {'politifact_fake': 400, 'politifact_real': 600, 'gossipcop_fake': 5300, 'gossipcop_real': 16000}
    if not os.path.exists(targetfolder):
        os.mkdir(targetfolder)
    dataset_generate(sourcefolder, targetfolder, sourcedataset_list, extract_num)