import ray
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import pandas as pd
import torch
import json
from pyspark.sql.functions import col, explode, lit, when
from datetime import datetime, timedelta
from transformers import AutoTokenizer, LlamaTokenizerFast, pipeline

# Initialize Ray with 2 actors
ray.init(ignore_reinit_error=True, 
         log_to_driver=False)

@ray.remote(num_gpus=1)
class TicketClassifier:
    def __init__(self, model_path: str):
        """
        Initialize the TicketClassifier with the specified model.

        Args:
            model_path (str): Path to the pre-trained model.
        """
        model_id = "microsoft/Phi-3-mini-4k-instruct" #"microsoft/Phi-3-mini-4k-instruct" #"microsoft/Phi-3-medium-4k-instruct"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto", 
            torch_dtype="float16", 
            trust_remote_code=True, 
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )
        self.llama_tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer") 
    
    def classify_ticket(self, ticket_text: str) -> Dict[str, str]:
        """
        Classify a support ticket based on its text content.

        Args:
            ticket_text (str): The text content of the support ticket.

        Returns:
            Dict[str, str]: A dictionary containing various classifications and analyses of the ticket.
        """
        generation_args = {
            "max_new_tokens": 700,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }
        if len(self.llama_tokenizer.encode(ticket_text))<20:
            return {k: "Not Generated" for k in [
                "Summary", "Category", "Main Issue", "Steps Taken", "Sentiment", "Urgency", 
                "Follow-Up Actions", "Repeated Issues", "Customer Loyalty Indicators", 
                "Transfer Involved", "Department Transferred To", "Issue Resolution Status", 
                "Satisfaction Score", "Improvement Suggestions"
            ]}
  
        system_prompt = f"""
        You are an AI assistant for the customer support team of PPL Electric Utilities.
        Your role is to analyze incoming customer support tickets and provide structured and detailed information to help our team respond quickly and effectively.

        Business Context:
        - We are an electrical utilities company based out of Pennsylvania serving customers in the PA region.  
        - We handle a variety of tickets daily, ranging from power outages to billing inquiries and renewable energy discussions.
        - Quick and accurate classification is crucial for customer satisfaction and operational efficiency.

        Your tasks:
        1. **Summarize the conversation:** Provide a concise summary of the ticket, highlighting key points discussed by the customer and the agent.
        2. **Categorize the ticket:** Assign the ticket to the most appropriate category from the following categories: Classify in the following categories: (BILLING/ BUSINESS ACCOUNTS/ COLLECTIONS/ COMMERCIAL/ ENERGY/ ON TRACK/ POWER/ STANDARD OFFER/ START OR STOP OR TRANSFER SERVICE)
        3. **Identify the main issue faced by the customer:** Clearly state the primary issue or concern raised by the customer.
        4. **Detail the steps taken to resolve the issue:** List the actions taken by the agent during the call to address the customer's concern.
        5. **Sentiment Analysis:** Analyze the emotional tone of the customer throughout the conversation.
        6. **Determine Urgency:** Assess the urgency of the issue based on the conversation. Classify in the following categories: (High/ Medium/ Low).
        7. **Identify any follow-up actions:** Detail any follow-up actions required or scheduled post-conversation.
        8. **Highlight any repeated issues:** Indicate if the issue discussed has been a repeated problem for the customer. Classify in the following categories: (Yes/No).
        9. **Customer loyalty indicators:** Evaluate if there are any indications of customer loyalty or dissatisfaction expressed during the call.
        10. **Transfer involved?:** Indicate if the call was transferred to another department. Classify in the following categories: (Yes/No).
        11. **Department transferred to:** If there was a transfer, specify the department to which the call was transferred. 
        12. **Issue resolution status:** State whether the customer's issue was resolved during the call (Resolved/Unresolved).
        13. **Satisfaction score for agent call handling:** Provide a score (1-5) assessing the agent's handling of the call, where 1 is very dissatisfied and 5 is very satisfied. Classify as (1/ 2/ 3/ 4/ 5) as output based on estimated satisfaction
        15. **Suggest Improvements in Call Handling:** Based on the analysis of the conversation, agent call handling skills and outcome of the call, suggest potential improvements to enhance efficiency and customer satisfaction.

        Analyze the following customer support ticket and provide the requested information in the specified format.

        Call: {ticket_text}

        Output Format:
        {{
            "Summary": <summary>,
            "Category": <category>,
            "Main Issue": <main issue>,
            "Steps Taken": <steps taken>,
            "Sentiment": <sentiment>,
            "Urgency": <urgency>,
            "Follow-Up Actions": <follow-up actions>,
            "Repeated Issues": <repeated issues Yes or No>,
            "Customer Loyalty Indicators": <customer loyalty indicators>,
            "Transfer Involved": <yes/no>,
            "Department Transferred To": <department>,
            "Issue Resolution Status": <resolved/unresolved>,
            "Satisfaction Score": <satisfaction score>
            "Improvement Suggestions": <improvement suggestions>
        }}

        Output:
        """
        messages = [
            {"role": "user", "content": system_prompt}
        ]  

        try:
            output = self.pipe(messages, **generation_args)
            return json.loads(output[0]['generated_text'])
        except Exception as e:
            logging.error(f"Error in classifying ticket: {str(e)}")
            return {k: "Not Generated" for k in [
                "Summary", "Category", "Main Issue", "Steps Taken", "Sentiment", "Urgency", 
                "Follow-Up Actions", "Repeated Issues", "Customer Loyalty Indicators", 
                "Transfer Involved", "Department Transferred To", "Issue Resolution Status", 
                "Satisfaction Score", "Improvement Suggestions"
            ]}
    
    def process_batch(self, batch: pd.DataFrame) -> List[Dict[str, str]]:
        """
        Process a batch of support tickets.

        Args:
            batch (pd.DataFrame): A DataFrame containing support tickets to process.

        Returns:
            List[Dict[str, str]]: A list of dictionaries, each containing classifications for a ticket.
        """
        results = []
        for _, row in batch.iterrows():
            result = self.classify_ticket(
                row['transcription']
            )
            results.append(result)
        return results        

def parallelize_inference(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Parallelize the inference process for a DataFrame of support tickets.

    Args:
        df (pd.DataFrame): DataFrame containing support tickets to process.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing classifications for a ticket.
    """
    BATCH_SIZE = 10  # Optimal batch size for efficient processing
    batches = [df.iloc[i:i + BATCH_SIZE] for i in range(0, len(df), BATCH_SIZE)]
    num_actors = 2  # Two actors available
    processors = [TicketClassifier.remote("microsoft/Phi-3-medium-4k-instruct") for _ in range(num_actors)]
    futures = [processors[i % num_actors].process_batch.remote(batch) for i, batch in enumerate(batches)]
    results = ray.get(futures)
    return [item for sublist in results for item in sublist]  # Flatten the list of results

def merger(anonymized_list: List[Dict[str, str]]) -> str:
    """
    Merge a list of anonymized transcriptions into a single string.

    Args:
        anonymized_list (List[Dict[str, str]]): A list of dictionaries containing anonymized transcriptions.

    Returns:
        str: A merged string of all transcriptions.
    """
    return " ".join([trans['text'] for trans in anonymized_list])

with open("/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/CallVoltMasterRepo/RepeatCallers/credentials/topics.json", "r") as f:
    creds = json.load(f)   

ip_path = creds['input_path']
transcription = pd.read_parquet(ip_path).iloc[:40, :]

transcription['transcription'] = transcription['anonymized'].apply(merger)

print("Number of Rows: "+str(len(transcription)))

result_list = parallelize_inference(transcription)

print(result_list)

# Convert result_list to DataFrame
output_df = pd.DataFrame(result_list)

#final['Satisfaction Score'] = final['Satisfaction Score'].apply(satisfaction_score_parser)
output_path = creds['output_path']
output_df.to_csv(output_path, index=False)
print(f"Output saved to {output_path}")

ray.shutdown()
