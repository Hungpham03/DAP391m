import numpy as np
import pandas as pd
import plotly.express as px

train_df = pd.read_excel("model_data/train_nor_811.xlsx", sheet_name=None)["Sheet1"]
valid_df = pd.read_excel("model_data/valid_nor_811.xlsx", sheet_name=None)["Sheet1"]
test_df = pd.read_excel("model_data/test_nor_811.xlsx", sheet_name=None)["Sheet1"]

train_df = train_df.drop("Unnamed: 0", axis=1)
valid_df = valid_df.drop("Unnamed: 0", axis=1)
test_df = test_df.drop("Unnamed: 0", axis=1)

train_df['Source'] = "Train"
valid_df['Source'] = "Valid"
test_df['Source'] = "Test"

df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

emotion_counts = df.groupby(["Source", "Emotion"])["Sentence"].count().reset_index()
emotion_counts.rename(columns={"Sentence": "Count"}, inplace=True)

order = ['Train', 'Valid', 'Test']
emotion_counts['Source'] = pd.Categorical(emotion_counts['Source'], categories=order, ordered=True)
emotion_counts.sort_values(by=['Source'], ascending=True, inplace=True)

fig = px.bar(emotion_counts, x='Emotion', y='Count', color='Source',
             labels={'Emotion': 'Emotion', 'Count': 'Count'},
             height=600)
fig.update_layout(barmode='stack')