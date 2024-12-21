import autogen
import json
import pandas as pd
from typing_extensions import Annotated
import streamlit as st


config_list = [
    {
        "model": "mixtral-8x7b-32768",
        "api_key": st.secrets["GROQ_API_KEY"],
        "api_type": "groq",
    }
]


llm_config = {"config_list": config_list, "temperature": 0}


# define agents
trust_evaluation_agent = autogen.UserProxyAgent(
    name="trust_evaluation_agent",
    system_message="You are a trust evaluator. Your goal is to gather feedbacks on the user trust perception in human-AI interaction from social psychologists to write report.",
    llm_config=llm_config,
    human_input_mode="NEVER",
    code_execution_config={
        "work_dir": "trust_evaluation",
        "use_docker": False,
    },
    max_consecutive_auto_reply=1,
)

competence = autogen.ConversableAgent(
    name="competence_trust_agent",
    llm_config=llm_config,
    system_message="You are a trained social psychologist, known for "
        "your ability to understand human users' language expression indicating their competence trust "
        "during human-AI interaction. "
        "You know that competence trust is about the user's belief in the AI system's ability, skills, and expertise "
        "to perform tasks effectively and accurately within its intended domain. "
        "You look for the following conversation signs of user trust: acceptance of advice, recognition of expertise, and follow-up questions. "
        "Based on your analysis, you need to give a score value from 1 to 7, where 1 indicates low competence trust "
        "You can rate 0 if the user does not show any signs of competence trust. "
        "Begin the analysis by stating your role.",
    human_input_mode="NEVER"
)

integrity = autogen.ConversableAgent(
    name="integrity_trust_agent",
    llm_config=llm_config,
    system_message="You are a trained social psychologist, known for "
        "your ability to understand human users' language expression indicating their integrity trust "
        "during human-AI interaction. "
        "You know that integrity trust is about the user's belief in that the AI system adheres to a set of acceptable principles, "
        "is honest about its capabilities and limitations, and provides truthful and accurate information. "
        "You look for the following conversation signs of user trust: questions about sources, verification requests, and acceptance of limitations. "
        "Based on your analysis, you need to give a score value from 1 to 7, where 1 indicates low integrity trust "
        "You can rate 0 if the user does not show any signs of integrity trust. "
        "Begin the analysis by stating your role.",
    human_input_mode="NEVER"
)

benevolence = autogen.ConversableAgent(
    name="benevolence_trust_agent",
    llm_config=llm_config,
    system_message="You are a trained social psychologist, known for "
        "your ability to understand human users' language expression indicating their benevolence trust "
        "during human-AI interaction. "
        "You know that benevolence trust is about the user's belief in that the AI system acts in their best interest, "
        "shows genuine concern for their needs, and aims to provide helpful and beneficial assistance. "
        "You look for the following conversation signs of user trust: personal disclosure, emotional sharing, and seeking guidance. "
        "Based on your analysis, you need to give a score value from 1 to 7, where 1 indicates low benevolence trust "
        "You can rate 0 if the user does not show any signs of benevolence trust. "
        "Begin the analysis by stating your role.",
    human_input_mode="NEVER"
)

predictability = autogen.ConversableAgent(
    name="predictability_trust_agent",
    llm_config=llm_config,
    system_message="You are a trained social psychologist, known for "
        "your ability to understand human users' language expression indicating their predictability trust "
        "during human-AI interaction. "
        "You know that predictability trust is about the user's belief in that the AI system's behaviors and responses are consistent, "
        "follow understandable patterns, and meet expected standards across interactions. "
        "You look for the following conversation signs of user trust: references to past interactions and expectations about responses. "
        "Based on your analysis, you need to give a score value from 1 to 7, where 1 indicates low predictability trust "
        "You can rate 0 if the user does not show any signs of predictability trust. "
        "Begin the analysis by stating your role.",
    human_input_mode="NEVER"
)

meta_assistant = autogen.AssistantAgent(
    name="meta_assistant",
    llm_config={
        # "cache_seed": 41,  # seed for caching and reproducibility
        "config_list": config_list,
        "temperature": 0,
    },
    code_execution_config=False,
    system_message="You are a meta assistant. Your role is to aggregate all feedbacks from social psychologists.",
    max_consecutive_auto_reply=1,
)


def add_trust_evaluation_to_csv(n_turn: str,
                    competence_trust_score: Annotated[int, "Rated score value for competence trust"], 
                    integrity_trust_score: Annotated[int, "Rated score value for integrity trust"], 
                    benevolence_trust_score: Annotated[int, "Rated score value for benevolence trust"], 
                    predictability_trust_score: Annotated[int, "Rated score value for predictability trust"], 
                    summary_text: Annotated[str, "Evaluation text"]):
    if n_turn == 1:
        # define dataframe
        df = pd.DataFrame(data={"n_turn": [], "competence_trust_score": [], "integrity_trust_score": [], "benevolence_trust_score": [], "predictability_trust_score": [], "summary_text": []})
    else:
        df = pd.read_csv("data/trust_evaluation.csv")
    # Add the new row
    new_row = [n_turn, competence_trust_score, integrity_trust_score, benevolence_trust_score, predictability_trust_score, summary_text]
    df.loc[len(df)] = new_row
    # Save the data
    df.to_csv("data/trust_evaluation.csv", index=False)
    # return df

    
def get_trust_evalution_data(conv_json):
    # Analyze each conversation turn    
    conversation = ""
    n_turn = 0
    for item in conv_json:
        print(item)
        conversation = conversation+item['role']+": "+item['content']+"\n"
        # each conversation turn is divided by user utterance
        if item['role'] == "user":
            n_turn += 1

            task_1 = """
            Review the conversation between human user and AI assistant.
            Rate the user trust dimension that you work for a score that is an integer from 1 to 7 and provide supporting evidence.
            The conversation content is as follows: {}
            """.format(conversation)

            # Agent team start trust evaluation
            chat_results = trust_evaluation_agent.initiate_chats(
                [
                    {
                        "recipient": competence,
                        "message": task_1,
                        "max_turns": 1,
                        "summary_method": "last_msg",
                    },
                    {
                        "recipient": integrity,
                        "message": task_1,
                        "max_turns": 1,
                        "summary_method": "last_msg",
                    },
                    {
                        "recipient": benevolence,
                        "message": task_1,
                        "max_turns": 1,
                        "summary_method": "last_msg",
                    },
                    {
                        "recipient": predictability,
                        "message": task_1,
                        "max_turns": 1,
                        "summary_method": "last_msg",
                    },
                    {
                        "recipient": meta_assistant,
                        "message": "Aggregate all feedbacks from social psychologists following the format: "
                        """{"competence_trust_score": , "integrity_trust_score": , "benevolence_trust_score": , "predictability_trust_score": , "summary_text": }.""",
                        "max_turns": 1,
                    },
                ]
            )

            output = chat_results[-1].chat_history[-1]["content"]
            ind_left_braces = output.index("{")
            ind_right_braces = output.index("}")
            output = output[ind_left_braces:ind_right_braces+1]
            output = json.loads(output)
            add_trust_evaluation_to_csv(n_turn, output["competence_trust_score"], output["integrity_trust_score"], output["benevolence_trust_score"], output["predictability_trust_score"], output["summary_text"])


