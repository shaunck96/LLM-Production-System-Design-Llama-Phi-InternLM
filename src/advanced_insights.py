import mlflow.pyfunc
import logging
import pandas as pd
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import time
import concurrent.futures
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class Adv(mlflow.pyfunc.PythonModel):
    """
    A custom MLflow PythonModel for advanced ticket classification using the Phi-3-mini-4k-instruct model.
    """
    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Load the model and tokenizer, and set up the text generation pipeline.

        Args:
            context (mlflow.pyfunc.PythonModelContext): The MLflow model context.
        """
        model_id = "microsoft/Phi-3-mini-4k-instruct"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer#,
            #batch_size=2
        )

    def classify_ticket(self, ticket_text: str) -> Dict[str, str]:
        """
        Classify a customer support ticket and extract various insights.

        Args:
            ticket_text (str): The text content of the customer support ticket.

        Returns:
            Dict[str, str]: A dictionary containing various classifications and insights about the ticket.
        """
        generation_args = {
            "max_new_tokens": 700,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }
        if len(self.tokenizer.encode(ticket_text)) < 20:
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
        14. **Suggest Improvements in Call Handling:** Based on the analysis of the conversation, agent call handling skills and outcome of the call, suggest potential improvements to enhance efficiency and customer satisfaction.

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
            "Satisfaction Score": <satisfaction score>,
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
    
    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict method required by MLflow's PythonModel interface.

        Args:
            context (mlflow.pyfunc.PythonModelContext): The MLflow model context.
            model_input (pd.DataFrame): Input data containing the 'text' column with the ticket content.

        Returns:
            Dict[str, Any]: Classification results for the input ticket.
        """
        return self.classify_ticket(model_input['text'].iloc[0])

def load_model_on_gpu() -> mlflow.pyfunc.PyFuncModel:
    """
    Load the Adv model on GPU and log it with MLflow.

    Returns:
        mlflow.pyfunc.PyFuncModel: The loaded MLflow model.
    """
    os.makedirs('adv', exist_ok=True)

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            artifact_path="adv",
            python_model=Adv(),
            artifacts={"adv": "./adv"}
        )

    model_uri = f"runs:/{run.info.run_id}/adv"
    return mlflow.pyfunc.load_model(model_uri)

def process_file(model: mlflow.pyfunc.PyFuncModel, transcription: str) -> Dict[str, Any]:
    """
    Process a single transcription file using the loaded model.

    Args:
        model (mlflow.pyfunc.PyFuncModel): The loaded MLflow model.
        transcription (str): The transcription text to process.

    Returns:
        Dict[str, Any]: A dictionary containing the original transcription, advanced insights, and processing time.
    """
    start_time = time.time()
    try:
        adv = model.predict(pd.DataFrame({'text': [transcription]}))
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        adv = {}
    processing_time = time.time() - start_time
    return {"transcription": transcription, "advanced_insights": adv, "processing_time": processing_time}

def process_files_parallel(model: mlflow.pyfunc.PyFuncModel, files: List[str], max_workers: int) -> List[Dict[str, Any]]:
    """
    Process multiple files in parallel using the loaded model.

    Args:
        model (mlflow.pyfunc.PyFuncModel): The loaded MLflow model.
        files (List[str]): List of transcription texts to process.
        max_workers (int): Maximum number of worker threads to use.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing results for each processed file.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, model, file) for file in files]
        return [future.result() for future in concurrent.futures.as_completed(futures)]

def merger(anonymized_list: List[Dict[str, Any]]) -> str:
    """
    Merge a list of anonymized transcriptions into a single string.

    Args:
        anonymized_list (List[Dict[str, Any]]): List of anonymized transcription dictionaries.

    Returns:
        str: Merged transcription text.
    """
    return " ".join([trans['text'] for trans in anonymized_list])

def load_credentials(filepath: str) -> Dict[str, str]:
    """
    Load credentials from a JSON file.

    Args:
        filepath (str): The path to the JSON file containing credentials.

    Returns:
        Dict[str, str]: A dictionary containing the loaded credentials.

    Raises:
        JSONDecodeError: If the file is not a valid JSON.
        FileNotFoundError: If the file is not found at the given path.
    """
    with open(filepath, "r") as f:
        return json.load(f)

def main_adv(files: List[str], model: mlflow.pyfunc.PyFuncModel, max_workers: int) -> pd.DataFrame:
    """
    Process a list of files using a machine learning model in parallel.

    Args:
        files (List[str]): A list of file contents to process.
        model (mlflow.pyfunc.PyFuncModel): The machine learning model to use for processing.
        max_workers (int): The maximum number of worker threads to use for parallel processing.

    Returns:
        pd.DataFrame: A DataFrame containing the results of processing all files.

    Raises:
        Exception: If an error occurs during processing. The error is logged, and an empty DataFrame is returned.
    """
    try:
        start_time = time.time()
        results = process_files_parallel(model, files, max_workers)
        final_df = pd.DataFrame(results)

        total_time = time.time() - start_time
        logger.info(f"Advanced Insights completed in {total_time:.2f} seconds.")
        logger.info(f"Total files evaluated: {len(final_df)}")

        return final_df
    
    except Exception as e:
        logger.error(f"Error in main_adv: {str(e)}")
        return pd.DataFrame()

def run_adv(max_workers: int) -> pd.DataFrame:
    """
    Run the advanced insights process on a set of transcriptions.

    This function loads a machine learning model, reads transcription data,
    processes it using the model, and saves the results.

    Args:
        max_workers (int): The maximum number of worker threads to use for parallel processing.

    Returns:
        pd.DataFrame: A DataFrame containing the results of processing all transcriptions.
                      Returns an empty DataFrame if an error occurs.

    Raises:
        Exception: If an error occurs during the process. The error is logged.
    """
    try:
        model = load_model_on_gpu()
        creds = load_credentials("/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/CallVoltMasterRepo/RepeatCallers/credentials/advanced_insights.json")   

        ip_path = creds['input_path']
        transcription = pd.read_parquet(ip_path)

        transcription['transcription'] = transcription['redacted'].apply(merger)
        file_names = transcription['transcription'].tolist()
        
        print(f"Number of Rows: {len(transcription)}")

        final_df = main_adv(file_names, model, max_workers)
        if not final_df.empty:
            output_path = creds['output_path']
            final_df.to_parquet(output_path)
            return final_df
        else:
            print("No results to display due to an error.")
    except Exception as e:
        logger.error(f"Error in run_adv: {str(e)}")

if __name__ == "__main__":
    max_workers = 2  # Adjust this based on your system's capabilities
    run_adv(max_workers)
