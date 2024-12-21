import pandas as pd
from convokit import PolitenessStrategies
import spacy

spacy_nlp = spacy.load('en_core_web_sm', disable=['ner'])
ps = PolitenessStrategies()

def rename_keys(dict):
    key_list = list(dict.keys())
    for key in key_list:
        key_new = key[21:]
        dict[key_new.replace("=", "")] = dict.pop(key)
    return dict

def dict_format_change_int_to_list(dict):
    for key in dict.keys():
        dict[key] = [dict[key]]
    return dict

def collect_politeness_strategies(n_turn, politeness_dict, utt_polite_markers, text):
    politeness_dict["n_turn"].append(n_turn)
    politeness_dict["text"].append(text)
    for key in utt_polite_markers.keys():
        politeness_dict[key].append(utt_polite_markers[key])
    return politeness_dict

def get_politeness_data(conv_json):
    n_turn = 0
    for item in conv_json:
        if item['role'] == "user":
            n_turn += 1
            if n_turn == 1:
                utt = ps.transform_utterance(item['content'], spacy_nlp=spacy_nlp)
                politeness_dict = utt.meta['politeness_strategies']
                politeness_dict = rename_keys(politeness_dict)
                politeness_dict = dict_format_change_int_to_list(politeness_dict)
                politeness_dict["n_turn"] = [n_turn]
                politeness_dict["text"] = [item['content']]
            else:
                utt = ps.transform_utterance(item['content'], spacy_nlp=spacy_nlp)
                utt_polite_markers = utt.meta['politeness_strategies']
                utt_polite_markers = rename_keys(utt_polite_markers)
                politeness_dict = collect_politeness_strategies(n_turn, politeness_dict, utt_polite_markers, item['content'])

    df_politeness_strategies = pd.DataFrame(data = politeness_dict)
    df_politeness_strategies.to_csv("data/politeness_strategies.csv", index=False)
