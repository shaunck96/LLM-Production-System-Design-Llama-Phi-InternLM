# Databricks notebook source
!pip install transformers==4.42.4
!pip install -U bitsandbytes
!pip install ray
!pip install pytest

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import ast 
import pandas as pd
from IPython.display import display, HTML
from jinja2 import Template

summaries = pd.read_parquet("/call_sid_based/benchmarking_results/gpu_transcriptions_redacted_summary_benchmarking.parquet")

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

def eval_func(transcription, summary):
    evaluation_prompt = """
    Given the following **call transcription** and **summary**, score the summary across various dimensions based on the provided transcription:
    
    Scoring Instructions:
    Accuracy: Does the summary capture all the key points and critical details of the call transcription? Ensure no false or misleading information is presented.
    Score from 1 to 10, where 1 means highly inaccurate, and 10 means the summary is completely accurate.

    Relevance: Does the summary focus on the most relevant and essential parts of the conversation? Irrelevant or minor details should be omitted.
    Score from 1 to 10, where 1 means irrelevant or off-topic, and 10 means only the most relevant parts are captured.

    Conciseness: Is the summary concise while still conveying the necessary information? Avoid unnecessary details or overly verbose sentences.
    Score from 1 to 10, where 1 means the summary is too verbose, and 10 means it's perfectly concise.

    Sentiment Alignment: Does the sentiment of the summary match the tone and emotional cues from the original call transcription? Ensure that positive, neutral, or negative sentiments are correctly represented.
    Score from 1 to 10, where 1 means the sentiment is misaligned, and 10 means the sentiment is perfectly aligned.
    
    Clarity: Is the summary easy to read and understand? Ensure that it is free from grammatical errors, unclear language, or awkward phrasing.
    Score from 1 to 10, where 1 means the summary is difficult to understand, and 10 means the summary is exceptionally clear.
    
    Overall Quality: Provide a holistic score that considers all the previous factors (accuracy, relevance, conciseness, sentiment, clarity).
    Score from 1 to 10, where 1 means poor overall quality, and 10 means excellent overall quality.

    The call transcription ad summary that needs to be evaluated are as follows:
    Call Transcription:
    {transcription}

    Summary:
    {summary}

    Required Output Format:
    {{
    "accuracy": {{
        "score": <score>
    }},
    "relevance": {{
        "score": <score>
    }},
    "conciseness": {{
        "score": <score>
    }},
    "sentiment_alignment": {{
        "score": <score>
    }},
    "clarity": {{
        "score": <score>
    }},
    "overall_quality": {{
        "score": <score>
    }}
    }}

    Example Output:
    {{
    "accuracy": {{
        "score": 9
    }},
    "relevance": {{
        "score": 8
    }},
    "conciseness": {{
        "score": 10
    }},
    "sentiment_alignment": {{
        "score": 7
    }},
    "clarity": {{
        "score": 9
    }},
    "overall_quality": {{
        "score": 8
    }}
    }}

    Only return the scoring json and no additional words in the output.

    Scoring Dictionary: 
    """.format(transcription=transcription, summary=summary)

    # Hello! How can I help you today?
    response, history = model.chat(tokenizer, evaluation_prompt)
    print(response)
    try:
        output = ast.literal_eval(response)  
        return output
    except:
        return {
            "accuracy": {
                "score": -1
            },
            "relevance": {
                "score": -1
            },
            "conciseness": {
                "score": -1
            },
            "sentiment_alignment": {
                "score": -1
            },
            "clarity": {
                "score": -1
            },
            "overall_quality": {
                "score": -1
            }
        }

# Applying the eval_func to each row in the DataFrame and expanding the dictionary into separate columns
summaries = summaries.iloc[:20, :]
summaries['eval'] = summaries.apply(lambda row: eval_func(row['transcription'], row['summary']), axis=1)

# Expanding the 'eval' dictionary into separate columns
eval_df = summaries['eval'].apply(pd.Series)

# Combining the evaluation scores with the original DataFrame
summaries = pd.concat([summaries, eval_df], axis=1)

summaries[['accuracy', 'relevance', 'conciseness', 'sentiment_alignment', 'clarity', 'overall_quality']] = summaries[['accuracy', 'relevance', 'conciseness', 'sentiment_alignment', 'clarity', 'overall_quality']].applymap(lambda x: x['score'] if isinstance(x, dict) else x)

# Assuming 'summaries' is your DataFrame with relevant columns
score_columns = ['accuracy', 'relevance', 'conciseness', 'sentiment_alignment', 'clarity', 'overall_quality']

# Function to generate insights with threshold-based comments
def generate_insight(metric_name, summary):
    mean = summary['mean']
    std = summary['std']
    min_value = summary['min']
    max_value = summary['max']
    
    # Default insight if no condition matches
    insight = f"Analysis for {metric_name} metric."

    # Set specific thresholds for metrics
    if metric_name == 'Accuracy':
        if mean > 0.8:
            insight = f"Excellent accuracy with a mean score of {mean:.2f}."
        elif mean > 0.6:
            insight = f"Moderate accuracy with a mean score of {mean:.2f}."
        else:
            insight = f"Low accuracy with a mean score of {mean:.2f}, indicating significant room for improvement."
    elif metric_name == 'Relevance':
        if mean > 0.7:
            insight = f"The relevance score is strong with an average of {mean:.2f}."
        else:
            insight = f"Relevance is moderate, averaging {mean:.2f}, indicating possible mismatch in some summaries."
    elif metric_name == 'Conciseness':
        if std < 0.5:
            insight = f"Conciseness is consistent with low variability (std = {std:.2f})."
        else:
            insight = f"Conciseness shows some variability (std = {std:.2f}), indicating inconsistency in summary length."
    elif metric_name == 'Sentiment Alignment':
        if mean > 0.75:
            insight = f"Strong sentiment alignment with a mean score of {mean:.2f}."
        else:
            insight = f"Sentiment alignment is moderate (mean = {mean:.2f}), suggesting room for improvement."
    elif metric_name == 'Clarity':
        if std < 1:
            insight = f"Clarity is consistent with low variability (std = {std:.2f}), indicating stable clarity across summaries."
        else:
            insight = f"Clarity shows significant variability (std = {std:.2f}), meaning some summaries may be unclear."
    elif metric_name == 'Overall Quality':
        if mean > 0.8:
            insight = f"Overall quality is excellent with a mean score of {mean:.2f}."
        elif mean > 0.6:
            insight = f"Overall quality is moderate with a mean score of {mean:.2f}."
        else:
            insight = f"Overall quality is poor with a mean score of {mean:.2f}, requiring significant improvement."

    return f"{insight} The scores range from {min_value} to {max_value} with a standard deviation of {std:.2f}."

# Create the dictionary to store summary statistics and insights for each metric
insights = {}
for column in score_columns:
    summary_stats = summaries[column].describe()
    insights[column] = generate_insight(column.capitalize(), summary_stats)

# HTML template to display the insights dynamically
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Summary Quality Metrics Report</title>
    <style>
        body { font-family: Arial, sans-serif; }
        h1 { text-align: center; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #dddddd; padding: 8px; text-align: center; }
        th { background-color: #f2f2f2; }
        .section-title { text-align: left; margin-top: 40px; }
    </style>
</head>
<body>

<h1>Summary Quality Metrics Report</h1>

{% for metric, insight in insights.items() %}
<h2 class="section-title">{{ loop.index }}. {{ metric.capitalize() }}</h2>
<p>{{ insight }}</p>
{% endfor %}

</body>
</html>
"""

# Use Jinja2 to render the template with the insights
template = Template(html_template)
rendered_html = template.render(insights=insights)

# Display the rendered HTML using IPython
display(HTML(rendered_html))

output_csv_path = '/summarization/summarization_metrics_evaluation.csv'
summaries.to_csv(output_csv_path)
