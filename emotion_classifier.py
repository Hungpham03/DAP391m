import numpy as np
import pandas as pd
from transformers import AutoTokenizer, RobertaForSequenceClassification
import torch

from preproc import preproc_class
from dist_list import emotion_dist

import plotly.express as px

model = RobertaForSequenceClassification.from_pretrained("model/", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("model/", local_files_only=True)

def find_key(input_dict, value):
    for key, val in input_dict.items():
        if val == value: return key
    return "None"

def perform_emotion_classification(input_text):
    input_ids = torch.tensor(
        [
            tokenizer.encode(
                preproc_class(lst=[input_text]).preprocessing([1, 2, 3, 4, 5, 6])[0]
            )
        ]
    )
    processed_text = preproc_class(lst=[input_text]).preprocessing([1, 2, 3, 4, 5, 6])[0]
    with torch.no_grad():
        out = model(input_ids)
        result = out.logits.softmax(dim=-1).tolist()

        predicted_emotion = find_key(emotion_dist, np.argmax(result[0]))
        confidence_score = f"{round(max(result[0])*100,2)}%"

    emotions = ["Enjoyment","Fear","Anger","Sadness", "Disgust","Surprise","Other"]
    percentages = [round(score * 100, 2) for score in result[0]]

    plot_data = {'Emotion': emotions, 'Percentage': percentages}
    df = pd.DataFrame(plot_data)

    color_map = {"Enjoyment": "#fa027e", "Fear": "#050500", "Anger": "#fc0303", "Sadness": "#03a9fc",  "Disgust": "#02fa23", "Surprise": "#fff200", "Other": "#191D88"}
    fig = px.bar(df, x='Emotion', y='Percentage', text='Percentage',
                 labels={'Emotion': 'Emotion', 'Percentage': 'Percentage'},
                 height=400, width=670, color='Emotion', color_discrete_map=color_map)
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')

    return predicted_emotion, confidence_score, fig, processed_text