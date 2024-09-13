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


def merger(anonymized_list):
    return " ".join([trans['text'] for trans in anonymized_list])

transcriptions = pd.read_csv(r"/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/Production_Code/transcriptions/gpu_adv_ins_call_sid_based.csv")
transcriptions


# Load the model (assuming you already have a compatible model setup)
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

# Define the evaluation function with scoring criteria for each transcription insight
def eval_func(transcription, category, main_issue, steps_taken, sentiment, urgency, follow_up_actions,
              repeated_issues, loyalty_indicators, transfer_involved, department_transferred_to, 
              resolution_status, satisfaction_score):
    
    evaluation_prompt = f"""
    Evaluate the quality of insights extracted from the following **transcription** across multiple dimensions:
    
    Transcription Insights:
    - **Category**: {category}
    - **Main Issue**: {main_issue}
    - **Steps Taken**: {steps_taken}
    - **Sentiment**: {sentiment}
    - **Urgency**: {urgency}
    - **Follow-Up Actions**: {follow_up_actions}
    - **Repeated Issues**: {repeated_issues}
    - **Customer Loyalty Indicators**: {loyalty_indicators}
    - **Transfer Involved**: {transfer_involved}
    - **Department Transferred To**: {department_transferred_to}
    - **Issue Resolution Status**: {resolution_status}
    - **Satisfaction Score**: {satisfaction_score}

    Scoring Instructions:
    
    1. **Category Accuracy**: Does the selected category accurately reflect the overall nature of the issue in the transcription?
       Score from 1 to 10.

    2. **Main Issue Relevance**: How well does the main issue capture the core problem in the transcription? Is the description clear and relevant?
       Score from 1 to 10.

    3. **Steps Taken Clarity**: Are the steps taken to resolve the issue clearly outlined? 
       Score from 1 to 10.
    
    4. **Sentiment Appropriateness**: Is the sentiment analysis accurate and appropriate based on the transcription content?
       Score from 1 to 10.

    5. **Urgency Detection**: How well does the system capture the urgency of the issue?
       Score from 1 to 10.

    6. **Follow-Up Actions**: How effectively are the follow-up actions identified and outlined in the transcription? 
       Score from 1 to 10.
    
    7. **Repeated Issues Identification**: Are repeated issues correctly detected in the transcription?
       Score from 1 to 10.

    8. **Customer Loyalty Indicators**: Are the customer loyalty indicators well-detected? Do they provide meaningful insights about the customer's relationship with the company?
       Score from 1 to 10.
    
    9. **Transfer Involvement Accuracy**: Is the presence of a transfer correctly identified? 
       Score from 1 to 10.

    10. **Department Transferred To**: Is the department the customer was transferred to correctly identified?
       Score from 1 to 10.

    11. **Issue Resolution Status**: Is the resolution status accurate? Does it match the actual outcome of the issue described in the transcription?
       Score from 1 to 10.

    12. **Satisfaction Score Accuracy**: Is the satisfaction score assigned to the customer accurate based on the transcription?
       Score from 1 to 10.

    13. **Overall Evaluation**: Based on all of the above criteria, provide an overall score for the transcription insight quality.
       Score from 1 to 10.

    Required Output Format:
    {{
      "category_accuracy": {{
          "score": <score>
      }},
      "main_issue_relevance": {{
          "score": <score>
      }},
      "steps_taken_clarity": {{
          "score": <score>
      }},
      "sentiment_appropriateness": {{
          "score": <score>
      }},
      "urgency_detection": {{
          "score": <score>
      }},
      "follow_up_actions": {{
          "score": <score>
      }},
      "repeated_issues_identification": {{
          "score": <score>
      }},
      "customer_loyalty_indicators": {{
          "score": <score>
      }},
      "transfer_involvement_accuracy": {{
          "score": <score>
      }},
      "department_transferred_to_accuracy": {{
          "score": <score>
      }},
      "issue_resolution_status": {{
          "score": <score>
      }},
      "satisfaction_score_accuracy": {{
          "score": <score>
      }},
      "overall_evaluation": {{
          "score": <score>
      }}
    }}

    Only return the scoring json and no additional words in the output.
    """
    
    # Call the model to generate the evaluation
    response, history = model.chat(tokenizer, evaluation_prompt)
    
    try:
        output = ast.literal_eval(response)  # Convert the response to a dictionary
        return output
    except:
        return {
            "category_accuracy": {"score": -1},
            "main_issue_relevance": {"score": -1},
            "steps_taken_clarity": {"score": -1},
            "sentiment_appropriateness": {"score": -1},
            "urgency_detection": {"score": -1},
            "follow_up_actions": {"score": -1},
            "repeated_issues_identification": {"score": -1},
            "customer_loyalty_indicators": {"score": -1},
            "transfer_involvement_accuracy": {"score": -1},
            "department_transferred_to_accuracy": {"score": -1},
            "issue_resolution_status": {"score": -1},
            "satisfaction_score_accuracy": {"score": -1},
            "overall_evaluation": {"score": -1}
        }

# Assuming you have a DataFrame 'transcriptions' with the appropriate columns
#transcriptions = transcriptions.iloc[:20, :]  # Example: Take the first 20 rows
transcriptions['eval'] = transcriptions.apply(lambda row: eval_func(row['transcription'], row['Category'], row['Main Issue'], row['Steps Taken'], 
                                                                    row['Sentiment'], row['Urgency'], row['Follow-Up Actions'], row['Repeated Issues'], 
                                                                    row['Customer Loyalty Indicators'], row['Transfer Involved'], row['Department Transferred To'], 
                                                                    row['Issue Resolution Status'], row['Satisfaction Score']), axis=1)

# Expanding the 'eval' dictionary into separate columns
eval_df = transcriptions['eval'].apply(pd.Series)

# Combining the evaluation scores with the original DataFrame
transcriptions = pd.concat([transcriptions, eval_df], axis=1)

# Extracting scores into individual columns
transcriptions[['category_accuracy', 'main_issue_relevance', 'steps_taken_clarity', 
                'sentiment_appropriateness', 'urgency_detection', 'follow_up_actions', 
                'repeated_issues_identification', 'customer_loyalty_indicators', 
                'transfer_involvement_accuracy', 'department_transferred_to_accuracy', 
                'issue_resolution_status', 'satisfaction_score_accuracy', 'overall_evaluation']] = transcriptions[
    ['category_accuracy', 'main_issue_relevance', 'steps_taken_clarity', 
     'sentiment_appropriateness', 'urgency_detection', 'follow_up_actions', 
     'repeated_issues_identification', 'customer_loyalty_indicators', 
     'transfer_involvement_accuracy', 'department_transferred_to_accuracy', 
     'issue_resolution_status', 'satisfaction_score_accuracy', 'overall_evaluation']].applymap(lambda x: x['score'] if isinstance(x, dict) else x)

# Summarize the metrics
summary = {
    "category_accuracy_avg": transcriptions['category_accuracy'].mean(),
    "main_issue_relevance_avg": transcriptions['main_issue_relevance'].mean(),
    "steps_taken_clarity_avg": transcriptions['steps_taken_clarity'].mean(),
    "sentiment_appropriateness_avg": transcriptions['sentiment_appropriateness'].mean(),
    "urgency_detection_avg": transcriptions['urgency_detection'].mean(),
    "follow_up_actions_avg": transcriptions['follow_up_actions'].mean(),
    "repeated_issues_identification_avg": transcriptions['repeated_issues_identification'].mean(),
    "customer_loyalty_indicators_avg": transcriptions['customer_loyalty_indicators'].mean(),
    "transfer_involvement_accuracy_avg": transcriptions['transfer_involvement_accuracy'].mean(),
    "department_transferred_to_accuracy_avg": transcriptions['department_transferred_to_accuracy'].mean(),
    "issue_resolution_status_avg": transcriptions['issue_resolution_status'].mean(),
    "satisfaction_score_accuracy_avg": transcriptions['satisfaction_score_accuracy'].mean(),
    "overall_evaluation_avg": transcriptions['overall_evaluation'].mean(),
}

# Step 2: Define an HTML template using Jinja2 with threshold-based coloring
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Call Insights Evaluation Summary</title>
    <style>
        body { font-family: Arial, sans-serif; }
        table { width: 80%; border-collapse: collapse; margin: 25px 0; }
        table, th, td { border: 1px solid black; padding: 8px; text-align: center; }
        th { background-color: #f2f2f2; }
        h2 { text-align: center; }
        .below-threshold { background-color: #ffcccc; } /* Red for below threshold */
        .above-threshold { background-color: #ccffcc; } /* Green for above threshold */
    </style>
</head>
<body>
    <h2>Call Insights Process Evaluation Summary</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Average Score</th>
        </tr>
        <tr>
            <td>Category Accuracy</td>
            <td>{{ category_accuracy_avg }}</td>
        </tr>
        <tr>
            <td>Main Issue Relevance</td>
            <td>{{ main_issue_relevance_avg }}</td>
        </tr>
        <tr>
            <td>Steps Taken Clarity</td>
            <td>{{ steps_taken_clarity_avg }}</td>
        </tr>
        <tr>
            <td>Sentiment Appropriateness</td>
            <td>{{ sentiment_appropriateness_avg }}</td>
        </tr>
        <tr>
            <td>Urgency Detection</td>
            <td>{{ urgency_detection_avg }}</td>
        </tr>
        <tr>
            <td>Follow-Up Actions</td>
            <td>{{ follow_up_actions_avg }}</td>
        </tr>
        <tr>
            <td>Repeated Issues Identification</td>
            <td>{{ repeated_issues_identification_avg }}</td>
        </tr>
        <tr>
            <td>Customer Loyalty Indicators</td>
            <td>{{ customer_loyalty_indicators_avg }}</td>
        </tr>
        <tr>
            <td>Transfer Involvement Accuracy</td>
            <td>{{ transfer_involvement_accuracy_avg }}</td>
        </tr>
        <tr>
            <td>Department Transferred To Accuracy</td>
            <td>{{ department_transferred_to_accuracy_avg }}</td>
        </tr>
        <tr>
            <td>Issue Resolution Status</td>
            <td>{{ issue_resolution_status_avg }}</td>
        </tr>
        <tr>
            <td>Satisfaction Score Accuracy</td>
            <td>{{ satisfaction_score_accuracy_avg }}</td>
        </tr>
        <tr>
            <td>Overall Evaluation</td>
            <td>{{ overall_evaluation_avg }}</td>
        </tr>
    </table>
</body>
</html>
"""

# Step 3: Render the template with the summary data
template = Template(html_template)
rendered_html = template.render(
    category_accuracy_avg=summary["category_accuracy_avg"],
    main_issue_relevance_avg=summary["main_issue_relevance_avg"],
    steps_taken_clarity_avg=summary["steps_taken_clarity_avg"],
    sentiment_appropriateness_avg=summary["sentiment_appropriateness_avg"],
    urgency_detection_avg=summary["urgency_detection_avg"],
    follow_up_actions_avg=summary["follow_up_actions_avg"],
    repeated_issues_identification_avg=summary["repeated_issues_identification_avg"],
    customer_loyalty_indicators_avg=summary["customer_loyalty_indicators_avg"],
    transfer_involvement_accuracy_avg=summary["transfer_involvement_accuracy_avg"],
    department_transferred_to_accuracy_avg=summary["department_transferred_to_accuracy_avg"],
    issue_resolution_status_avg=summary["issue_resolution_status_avg"],
    satisfaction_score_accuracy_avg=summary["satisfaction_score_accuracy_avg"],
    overall_evaluation_avg=summary["overall_evaluation_avg"]
)

# Step 4: Save the rendered HTML to a file
output_path = "/Workspace/Users/sshibu@pplweb.com/CS_PA_NLP/call_insights_summary.html"
with open(output_path, "w") as f:
    f.write(rendered_html)

# Optionally: Display the rendered HTML in a Jupyter environment
from IPython.core.display import display, HTML
display(HTML(rendered_html))

output_csv_path = '/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/CallVoltMasterRepo/RepeatCallers/evaluation/advanced_insights/advanced_insights_metrics_evaluation.csv'
summary.to_csv(output_csv_path)
