from transformers import pipeline, AutoTokenizer


instruction = ("I need your assistance in evaluating the authenticity of a news article. "
               "I will provide you the news article and additional information about this news. "
               "You have to answer that [This is fake news] or [This is real news]"
               " in the first sentence of your output and give your explanation about [target news].\n")

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
def prompt_generate(news, dem=None, google_use=False):
    news_text, news_tweet = news_cut(news, n=100, m=50)
    input_news = "news title: {}, news text: {}, news tweet: {}".format(
        news['title'], news_text, news_tweet)

    content = ("    [target news]:\n"
               "        [input news]: [{}]\n"
               "        [output]: ") .format(input_news)

    demostration = ('I will give you some examples of news, '
                    'Your answer after [output] should be consistent with the following examples:\n')
    if dem is not None:
        for i, dem_news in enumerate(dem):
            dem_news_text, dem_news_tweet = news_cut(dem_news, n=100, m=50)
            input_example = "news title: {}, news text: {}, news tweet: {}".format(
                dem_news['title'], dem_news_text, dem_news_tweet,)
            demostration += ("    [example {}]:\n"
                             "        [input news]: [{}]\n"
                             "        [output]: [This is {} news]\n").format(i + 1,
                                                                             input_example,
                                                                             dem_news['label'])
    if google_use:
        google_result = ("I will give you additional information about this news: {}".format(news['google']))
        return instruction + google_result + demostration + content

    return instruction + demostration + content



