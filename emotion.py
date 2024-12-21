from transformers import pipeline
import re
import pandas as pd


def split_sentences(user_prompt):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', user_prompt)
    return sentences

def get_emotion_data(conv_json):
    n_turn = 0
    user_prompt_dict = {}
    for item in conv_json:
        if item['role'] == "user":
            n_turn += 1
            sentences = split_sentences(item['content'])
            for sent in sentences:
                user_prompt_dict[sent] = n_turn
    user_prompt_message_list = list(user_prompt_dict.keys())
    classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True)
    prediction = classifier(user_prompt_message_list)
    prediction_result = {'text':[], 'n_turn':[], 'emotion':[], 'score':[]}
    for i in range(len(user_prompt_message_list)):
        for sub_emo_dict in prediction[i]:
            prediction_result['text'].append(user_prompt_message_list[i])
            prediction_result['n_turn'].append(user_prompt_dict[user_prompt_message_list[i]])
            prediction_result['emotion'].append(sub_emo_dict['label'])
            prediction_result['score'].append(sub_emo_dict['score'])

    df = pd.DataFrame(prediction_result)
    df.drop('text', axis=1, inplace=True)
    df = df.groupby(['n_turn', 'emotion']).mean().reset_index()
    df['z-score'] = (df['score'] - df['score'].mean())/df['score'].std()
    df.to_csv('data/user_prompt_emotion.csv', index=False)
