from nltk.tokenize import RegexpTokenizer
from nltk.corpus import brown
from nltk.probability import FreqDist
import pandas as pd
import numpy as np
import re

# load brown corpus 
brown_words = brown.words()
fdist = FreqDist(brown_words)
total_word_count = sum(fdist.values())


def split_sentences(user_prompt):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', user_prompt)
    return sentences

def response_length(sentences):
    word_count_by_sent = []
    for sent in sentences:
        word_list = sent.split()
        word_count_by_sent.append(len(word_list))
    return sum(word_count_by_sent)

def add_user_engagement_data(n_turn, response_len, informativeness):
    if n_turn == 1:
        # define dataframe
        df = pd.DataFrame(columns=['n_turn', 'response_length', 'informativeness'])
    else:
        df = pd.read_csv("data/user_engagement.csv")
    new_row = [n_turn, response_len, informativeness]
    df.loc[len(df)] = new_row
    df.to_csv("data/user_engagement.csv", index=False)

# check word count from brown corpus
def word_count_brown_corpus(word_token, total_word_count=total_word_count):
    return float(fdist[word_token]/total_word_count)

# get tokens without punctuation
def get_unique_word_tokens(sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence.lower())
    return tokens

def informativeness(sentences):
    word_list = []
    for sent in sentences:
        tokens = get_unique_word_tokens(sent)
        word_list.extend(tokens)
    word_list = set(word_list)
    informtv = []
    try:
        for word in word_list:
            informtv.append(np.log2(1/word_count_brown_corpus(word)))
    except ZeroDivisionError:
        pass
    return sum(informtv)

def get_user_engagement_data(conv_json):
    n_turn = 0
    for item in conv_json:
        if item['role'] == "user":
            n_turn += 1

            # user engegement metrics
            user_prompt = item['content']
            sentences = split_sentences(user_prompt) 
            response_len = response_length(sentences)
            informatv = informativeness(sentences)
            add_user_engagement_data(n_turn, response_len, informatv)