
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
# import seaborn as sns
# import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events


# load trust eveluation data 
df_trust = pd.read_csv('data/trust_evaluation.csv')
# row = df_trust.iloc[0,:4] 
# trust_data = pd.Series(row.to_list(), index=row.index)

# load user engagement data
df_engagement = pd.read_csv('data/user_engagement.csv')

# load politeness strategies data
df_politeness_strategies = pd.read_csv('data/politeness_strategies.csv')

# load user prompt emotion data
df_emotion = pd.read_csv('data/user_prompt_emotion.csv')

# functions for visualization
def normalize_columns(df, columns_to_normalize):
    for column in columns_to_normalize:
        # Create a scaler object
        scaler = MinMaxScaler()
        # Fit the scaler to the data and transform it
        df[[column]] = scaler.fit_transform(df[[column]])
    return df

def heat_map_data(df):
    user_prompts = df["text"]
    df.drop(['text', 'n_turn'], axis=1, inplace=True)
    transpose_df = df.T
    transpose_df = transpose_df.reset_index()
    transpose_df.rename(columns={'index': 'Politeness_Strategies'}, inplace=True)
    conversation_turns = transpose_df.columns
    politeness_strategies = transpose_df["Politeness_Strategies"]
    polite_markers_color_matrix = transpose_df.iloc[:,1:].values
    return conversation_turns, user_prompts, politeness_strategies, polite_markers_color_matrix


# normalize the data
# df_trust = normalize_columns(df_trust, ["competence_trust", "benevolence_trust", "integrity_trust", "predictability_trust"])
df_trust.loc[:,["competence_trust_score", "benevolence_trust_score", "integrity_trust_score", "predictability_trust_score"]] = df_trust.loc[:,["competence_trust_score", "benevolence_trust_score", "integrity_trust_score", "predictability_trust_score"]].apply(lambda x: (x-0)/ (7-0), axis=0)
df_engagement = normalize_columns(df_engagement, ["response_length", "informativeness"])

# set page layout
st.set_page_config(
    layout="wide",
)

# main visualization section 
col1, col2 = st.columns([1, 1])
with col1:
    # trust evaluation over conversation turns

    # trust_dict = df_trust.to_dict('list') 
    # Create the bar chart
    # fig = px.bar(trust_data)
    # Display the figure in Streamlit
    # st.plotly_chart(fig, x='Trust dimension', y='Score')

    # Create figure
    fig_1 = go.Figure()
    
    # Add traces
    fig_1.add_trace(go.Scatter(x=df_trust['n_turn'].astype(str), y=df_trust['competence_trust_score'], name='Competence Trust', 
                               line_color='#1976d2', mode='lines+markers', opacity=0.8, marker=dict(opacity=1)))
    # fig_1.update_traces(hoverinfo=df_trust['summary_text'])
    fig_1.add_trace(go.Scatter(x=df_trust['n_turn'].astype(str), y=df_trust['benevolence_trust_score'], name='Benevolence Trust', 
                               line_color="#fbc02d", mode='lines+markers', opacity=0.8, marker=dict(opacity=1)))
    fig_1.add_trace(go.Scatter(x=df_trust['n_turn'].astype(str), y=df_trust['integrity_trust_score'], name='Integrity Trust', 
                               line_color='#66bb6a', mode='lines+markers', opacity=0.8, marker=dict(opacity=1)))
    fig_1.add_trace(go.Scatter(x=df_trust['n_turn'].astype(str), y=df_trust['predictability_trust_score'], name='Predictability Trust', 
                               line_color='#f06292', mode='lines+markers', opacity=0.8, marker=dict(opacity=1)))

    # Update layout
    fig_1.update_layout(title='User Trust Development in Human-AI Communication', xaxis_title='Conversation Turns', yaxis_title='Trust Perception')
    fig_1.update_layout(autosize=False, width=590, height=340)
    # st.plotly_chart(fig_1, key="fig_1")
    selected_points_1 = plotly_events(fig_1, override_height=345)

    if selected_points_1:
        txt = st.text_area(
            "**Evaluation summary:**",
            df_trust['summary_text'].to_list()[selected_points_1[0]["pointIndex"]],
            height=270
        )

    # politeness strategies over conversation turns 
    conversation_turns, user_prompts, politeness_strategies, polite_markers_color_matrix = heat_map_data(df_politeness_strategies)
    # Create the heat map
    fig_3 = go.Figure(
        data=go.Heatmap(
            z=polite_markers_color_matrix,  # Color intensity matrix
            x=conversation_turns,  # Conversation turns
            y=politeness_strategies,  # Categories
            colorscale='greys',  # Color scale
        )
    )

    fig_3.update_traces(showscale=False)
    # Update layout
    fig_3.update_layout(
        title="User Theory of Politeness in Human-AI Communication",
        title_x=0.2,
        xaxis_title="Conversation Turns",
        yaxis_title="Politeness Strategies",
        yaxis=dict(categoryorder="array", categoryarray=politeness_strategies)   
    )
    fig_3.update_layout(autosize=True, width=590, height=540)
    st.write(fig_3)

with col2:
    # user engagement over conversation turns

    # trust_dict = df_trust.to_dict('list') 
    # Create the bar chart
    # fig = px.bar(trust_data)
    # Display the figure in Streamlit
    # st.plotly_chart(fig, x='Trust dimension', y='Score')

    # Create figure
    fig_2 = go.Figure()
    
    # Add traces
    fig_2.add_trace(go.Scatter(x=df_engagement['n_turn'].astype(str), y=df_engagement['response_length'], name='Response Length', 
                               line_color='#1976d2', mode='lines+markers', opacity=0.8, marker=dict(opacity=1)))
    fig_2.add_trace(go.Scatter(x=df_engagement['n_turn'].astype(str), y=df_engagement['informativeness'], name='Informativeness',
                                line_color="#fbc02d", mode='lines+markers', opacity=0.8, marker=dict(opacity=1)))
    # fig_2.add_trace(go.Scatter(x=df_trust['n_turn'].astype(str), y=df_trust['integrity_trust'], name='Integrity Trust', line_color='#66bb6a'))
    # fig_2.add_trace(go.Scatter(x=df_trust['n_turn'].astype(str), y=df_trust['predictability_trust'], name='Predictability Trust', line_color='#f06292'))


    # Update layout (optional)
    fig_2.update_layout(title='User Engagement in Human-AI Communication', xaxis_title='Conversation Turns', yaxis_title='Metric Measurement')
    fig_2.update_layout(autosize=False, width=620, height=360)
    # st.plotly_chart(fig_2, key="fig_2")
    selected_points_2 = plotly_events(fig_2, override_height=365)

    # user prompt emotion over conversation turns
    # Create figure
    fig_4 = px.bar(df_emotion, x = "n_turn", y = "z-score",
             color = "emotion")
    fig_4.update_layout(
        title="User Emotion in Human-AI Communication",
        title_x=0.2,
        xaxis_title="Conversation Turns",
        # yaxis_title="Politeness Strategies",
        # yaxis=dict(categoryorder="array", categoryarray=politeness_strategies)   
    )
    fig_3.update_layout(autosize=True, width=530, height=370)
    st.write(fig_4)


