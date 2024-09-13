# Databricks notebook source
!pip install transformers==4.42.4
!pip install -U bitsandbytes
!pip install ray
!pip install pytest

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import ast 
from jinja2 import Template
import json

topics = pd.read_parquet(r"/gpu_transcriptions_redacted_topics_2024_07_31.parquet")

with open(r"/Workspace/Users/sshibu@pplweb.com/CS_PA_NLP/mp3_files/topics/topic_og.json", "r") as f:
    json_data = f.read()

    data = json.loads(json_data)

master_list_of_topics = ', '.join(list(data.keys()))

model_path = "internlm/internlm2_5-7b-chat"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32, 
    device_map="auto",
    trust_remote_code=True, 
    load_in_4bit=True  # Use 4-bit quantization
).eval()
tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    trust_remote_code=True
)

def eval_func(transcription_summary, relevant_topics, all_available_topics):
    evaluation_prompt = """
    Given the following **transcription summary** and **list of relevant topics**, evaluate the quality of topic classification for this call. The system should correctly identify relevant topics and avoid incorrect classifications.

    Scoring Instructions:
    
    **Topic Relevance**: Does the list of selected topics accurately reflect the main themes and content of the transcription summary? Are the most important topics captured effectively?
    Score from 1 to 10, where 1 means poor relevance (many important topics missed or irrelevant topics included) and 10 means perfect relevance (all important topics captured, no irrelevant topics included).

    **Topic Coverage**: Does the list of selected topics cover the main points of the call, ensuring no major aspect of the conversation is missed? 
    Score from 1 to 10, where 1 means poor coverage (important points missed) and 10 means perfect coverage (all key points captured).

    **Topic Accuracy**: Are the selected topics correct in the context of the call summary? This evaluates if there are any topics that are incorrectly included or classified.
    Score from 1 to 10, where 1 means low accuracy (many incorrect topics) and 10 means high accuracy (all topics correct).

    **Classification Consistency**: Are similar topics consistently identified across multiple transcriptions of similar nature? Does the system classify calls in a consistent manner across calls with similar themes?
    Score from 1 to 10, where 1 means poor consistency (topics vary significantly for similar content) and 10 means high consistency (topics are consistently identified).

    **Call Classification Speed**: Based on the assumed automated process, evaluate the efficiency and speed of the classification process. This is based on overall execution and expected runtime.
    Score from 1 to 10, where 1 means very slow or inefficient, and 10 means fast and highly efficient.

    **Overall Classification Quality**: Provide a holistic score that considers all the previous factors (topic relevance, coverage, accuracy, consistency, speed).
    Score from 1 to 10, where 1 means poor overall classification quality, and 10 means excellent overall classification quality.

    The transcription summary and relevant topics to be evaluated are as follows:
    
    Transcription Summary:
    {transcription_summary}

    Relevant Topics:
    {relevant_topics}

    Available Topics:
    {all_available_topics}

    Required Output Format:
    {{
    "topic_relevance": {{
        "score": <score>
    }},
    "topic_coverage": {{
        "score": <score>
    }},
    "topic_accuracy": {{
        "score": <score>
    }},
    "classification_consistency": {{
        "score": <score>
    }},
    "call_classification_speed": {{
        "score": <score>
    }},
    "overall_classification_quality": {{
        "score": <score>
    }}
    }}

    Only return the scoring json and no additional words in the output.
    """.format(transcription_summary=transcription_summary, relevant_topics=relevant_topics, all_available_topics=all_available_topics)

    # Call the model to generate the evaluation
    response, history = model.chat(tokenizer, evaluation_prompt)
    print(response)
    try:
        output = ast.literal_eval(response)  # Convert the response to a dictionary
        return output
    except:
        return {
            "topic_relevance": {
                "score": -1
            },
            "topic_coverage": {
                "score": -1
            },
            "topic_accuracy": {
                "score": -1
            },
            "classification_consistency": {
                "score": -1
            },
            "call_classification_speed": {
                "score": -1
            },
            "overall_classification_quality": {
                "score": -1
            }
        }

topics = topics.iloc[:20, :]
topics['eval'] = topics.apply(lambda row: eval_func(row['summary'], row['topics'], master_list_of_topics), axis=1)

# Expanding the 'eval' dictionary into separate columns
eval_df = topics['eval'].apply(pd.Series)

# Combining the evaluation scores with the original DataFrame
topics = pd.concat([topics, eval_df], axis=1)

# Extracting scores into individual columns
topics[['topic_relevance', 'topic_coverage', 'topic_accuracy', 'classification_consistency', 'call_classification_speed', 'overall_classification_quality']] = topics[['topic_relevance', 'topic_coverage', 'topic_accuracy', 'classification_consistency', 'call_classification_speed', 'overall_classification_quality']].applymap(lambda x: x['score'] if isinstance(x, dict) else x)

# Summarize the metrics
summary = {
    "topic_relevance_avg": topics['topic_relevance'].mean(),
    "topic_coverage_avg": topics['topic_coverage'].mean(),
    "topic_accuracy_avg": topics['topic_accuracy'].mean(),
    "classification_consistency_avg": topics['classification_consistency'].mean(),
    "call_classification_speed_avg": topics['call_classification_speed'].mean(),
    "overall_classification_quality_avg": topics['overall_classification_quality'].mean(),
}

# Define thresholds for each metric
thresholds = {
    "topic_relevance": 7,
    "topic_coverage": 7,
    "topic_accuracy": 7,
    "classification_consistency": 7,
    "call_classification_speed": 7,
    "overall_classification_quality": 7,
}

# Summarize the metrics
summary = {
    "topic_relevance_avg": topics['topic_relevance'].mean(),
    "topic_coverage_avg": topics['topic_coverage'].mean(),
    "topic_accuracy_avg": topics['topic_accuracy'].mean(),
    "classification_consistency_avg": topics['classification_consistency'].mean(),
    "call_classification_speed_avg": topics['call_classification_speed'].mean(),
    "overall_classification_quality_avg": topics['overall_classification_quality'].mean(),
}

# Step 2: Define an HTML template using Jinja2 with threshold-based coloring
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Call Classification Evaluation Summary</title>
    <style>
        body { font-family: Arial, sans-serif; }
        table { width: 50%; border-collapse: collapse; margin: 25px 0; }
        table, th, td { border: 1px solid black; padding: 8px; text-align: center; }
        th { background-color: #f2f2f2; }
        h2 { text-align: center; }
        .below-threshold { background-color: #ffcccc; } /* Red for below threshold */
        .above-threshold { background-color: #ccffcc; } /* Green for above threshold */
    </style>
</head>
<body>
    <h2>Call Classification Process Evaluation Summary</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Average Score</th>
        </tr>
        <tr class="{{ 'below-threshold' if topic_relevance_avg < thresholds['topic_relevance'] else 'above-threshold' }}">
            <td>Topic Relevance</td>
            <td>{{ topic_relevance_avg }}</td>
        </tr>
        <tr class="{{ 'below-threshold' if topic_coverage_avg < thresholds['topic_coverage'] else 'above-threshold' }}">
            <td>Topic Coverage</td>
            <td>{{ topic_coverage_avg }}</td>
        </tr>
        <tr class="{{ 'below-threshold' if topic_accuracy_avg < thresholds['topic_accuracy'] else 'above-threshold' }}">
            <td>Topic Accuracy</td>
            <td>{{ topic_accuracy_avg }}</td>
        </tr>
        <tr class="{{ 'below-threshold' if classification_consistency_avg < thresholds['classification_consistency'] else 'above-threshold' }}">
            <td>Classification Consistency</td>
            <td>{{ classification_consistency_avg }}</td>
        </tr>
        <tr class="{{ 'below-threshold' if call_classification_speed_avg < thresholds['call_classification_speed'] else 'above-threshold' }}">
            <td>Call Classification Speed</td>
            <td>{{ call_classification_speed_avg }}</td>
        </tr>
        <tr class="{{ 'below-threshold' if overall_classification_quality_avg < thresholds['overall_classification_quality'] else 'above-threshold' }}">
            <td>Overall Classification Quality</td>
            <td>{{ overall_classification_quality_avg }}</td>
        </tr>
    </table>
</body>
</html>
"""

# Step 3: Render the template with the summary data
template = Template(html_template)
rendered_html = template.render(
    topic_relevance_avg=summary["topic_relevance_avg"],
    topic_coverage_avg=summary["topic_coverage_avg"],
    topic_accuracy_avg=summary["topic_accuracy_avg"],
    classification_consistency_avg=summary["classification_consistency_avg"],
    call_classification_speed_avg=summary["call_classification_speed_avg"],
    overall_classification_quality_avg=summary["overall_classification_quality_avg"],
    thresholds=thresholds  # Pass the threshold values to the template
)

# Step 4: Save the rendered HTML to a file
output_path = "/Workspace/Users/sshibu@pplweb.com/CS_PA_NLP/call_classification_summary.html"
with open(output_path, "w") as f:
    f.write(rendered_html)

# Optionally: Display the rendered HTML in a Jupyter environment
from IPython.core.display import display, HTML
display(HTML(rendered_html))

output_csv_path = '/evaluation/topic_modelling/topic_modelling_metrics_evaluation.csv'
summary.to_csv(output_csv_path)
