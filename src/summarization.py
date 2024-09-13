import os
import time
import json
from typing import List, Dict, Any, Optional
import pandas as pd
import mlflow
import mlflow.pyfunc
import nltk
from nltk.tokenize import word_tokenize
from langchain.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, LlamaTokenizerFast
# Download NLTK data
nltk.download('punkt')


class LlamaCppModel(mlflow.pyfunc.PythonModel):
    """A wrapper class for LlamaCpp model to be used with MLflow."""
    def __init__(self, llama_cpp_model: LlamaCpp):
        """
        Initialize the LlamaCppModel.

        Args:
            llama_cpp_model (LlamaCpp): An instance of LlamaCpp model.
        """
        self.llama_cpp_model = llama_cpp_model
 
    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame) -> str:
        """
        Make predictions using the LlamaCpp model.

        Args:
            context (mlflow.pyfunc.PythonModelContext): The model context.
            model_input (pd.DataFrame): Input data for prediction.

        Returns:
            str: The model's prediction.
        """
        output = self.llama_cpp_model.predict(model_input)
        return output
 
def _load_pyfunc(llm: LlamaCpp) -> LlamaCppModel:
    """
    Create a LlamaCppModel instance.

    Args:
        llm (LlamaCpp): An instance of LlamaCpp model.

    Returns:
        LlamaCppModel: A wrapped LlamaCpp model.
    """
    return LlamaCppModel(llm)

def setup_llama() -> mlflow.pyfunc.PyFuncModel:
    """
    Set up and return a LlamaCpp model wrapped in MLflow.

    Returns:
        mlflow.pyfunc.PyFuncModel: A loaded MLflow model.
    """
    llama_config = {
        'lower_limit': 200,
        'upper_limit': 500,
        'gpu_layers': -1,
        'n_batch': 512,
        'num_actors': 2,
        'batch_size': 10,
        'advtransformer_model': "TheBloke/Llama-2-7B-Chat-GGUF",
        'advtransformer_basename': "llama-2-7b-chat.Q5_K_S.gguf",
        'advtokenizer_model': "hf-internal-testing/llama-tokenizer",
        'active': True
    }
    
    model_name_or_path = llama_config['advtransformer_model']
    model_basename = llama_config['advtransformer_basename']
    model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
    
    n_gpu_layers = llama_config['gpu_layers'] # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
    n_batch = llama_config['n_batch']  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

    # Make sure the model path is correct for your system!
    llm_model = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        #verbose=True,  # Verbose is required to pass to the callback manager
        n_ctx=4096
    )
    
    model_instance = _load_pyfunc(llm_model)
    
    os.makedirs('llama_cpp_model', exist_ok=True)
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            artifact_path="llama_cpp_model",
            python_model=model_instance,
            artifacts={"llama_cpp_model": "./llama_cpp_model"}
            # signature=infer_signature(model_input),
            # input_example=pd.DataFrame(model_input, columns=['text'])
        )
    
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/llama_cpp_model"
        print(f"Logged model URI: {model_uri}")

    return model_uri
 
class LlamaSummarizer:
    """A class for summarizing text using LlamaCpp model."""
    def __init__(self, model_uri: str):
        """
        Initialize the LlamaSummarizer.

        Args:
            model_uri (str): The URI of the MLflow model to load.
        """
        self.llm = mlflow.pyfunc.load_model(model_uri)
        self.tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
 
    def custom_text_splitter(self, text: str, chunk_size: int = 3200, max_tokens: int = 3800, overlap: int = 50) -> List[str]:
        """
        Split the input text into chunks.

        Args:
            text (str): The input text to split.
            chunk_size (int): The size of each chunk in tokens.
            max_tokens (int): The maximum number of tokens allowed in a chunk.
            overlap (int): The number of overlapping tokens between chunks.

        Returns:
            List[str]: A list of text chunks.
        """
        tokens = self.tokenizer.encode(text)
        start = 0
        chunks = []
 
        while start < len(tokens):
            end = start + chunk_size
           
            # Ensure we do not exceed the token limit after adding overlap
            if end + overlap < len(tokens):
                end += overlap
           
            # Make sure the end does not exceed the maximum token limit
            if end > len(tokens):
                end = len(tokens)
            if (end - start) > max_tokens:
                end = start + max_tokens
 
            # Decode the tokens back to text and add to chunks
            chunk_text = self.tokenizer.decode(tokens[start:end])
            chunks.append(chunk_text)
           
            # Move start forward by chunk size, ensuring it does not overlap beyond max_tokens
            start = end - overlap if (end - overlap > start + chunk_size) else start + chunk_size
           
        return chunks
   
    def merger(self, anonymized_list: List[Dict[str, str]]) -> str:
        """
        Merge a list of anonymized transcriptions into a single string.

        Args:
            anonymized_list (List[Dict[str, str]]): A list of dictionaries containing anonymized text.

        Returns:
            str: A merged string of all transcriptions.
        """
        return " ".join([trans['text'] for trans in anonymized_list])
 
    def word_count(self, text: str) -> int:
        """
        Count the number of words in a given text.

        Args:
            text (str): The input text.

        Returns:
            int: The number of words in the text.
        """
        return len(word_tokenize(text))
 
    def summary(self, text: str, token_count: int) -> str:
        """
        Generate a summary of the input text.

        Args:
            text (str): The input text to summarize.
            token_count (int): The number of tokens in the input text.

        Returns:
            str: A summary of the input text.
        """
        prompt = """
    [INST] <<SYS>> Summarize the key discussion points between the customer and agent in under 150 words.
    <<SYS>> {} [/INST]
    """
        if token_count>3900:
            chunks = self.custom_text_splitter(text)
            chunk_summaries = []
            for chunk in chunks:
                summary = self.llm.predict(prompt.format(chunk))
                chunk_summaries.append(summary)
            combined_summary = "\n".join(chunk_summaries)
            return combined_summary
        else:
            return(self.llm.predict(prompt.format(text)))
 
    def actions_taken(self, text: str, token_count: int) -> str:
        """
        Identify actions taken in the conversation.

        Args:
            text (str): The input text to analyze.
            token_count (int): The number of tokens in the input text.

        Returns:
            str: A description of actions taken or "No actions generated" if token count is too high.
        """
        if token_count<3900:
            prompt = f"[INST] <<SYS>> For each conversation, analyze the interaction between the customer and the agent. Identify and return a detailed list of the specific steps or actions taken by the agent to address and resolve the customer's query or issue in under 150 words.<<SYS>> {text} [/INST]"
            return self.llm.predict(prompt)
        else:
            return "No actions generated"
 
    def summary_split(self, text: str, token_count: int) -> str:
        """
        Generate a summary of the input text, splitting if necessary.

        Args:
            text (str): The input text to summarize.
            token_count (int): The number of tokens in the input text.

        Returns:
            str: A summary of the input text.
        """
        # Helper function to process text chunks with the LLM
        def process_chunk(text_chunk: str) -> str:
            prompt = f"""
            [INST] <<SYS>> Summarize the key discussion points between the customer and agent in under 150 words.
            <<SYS>> {text_chunk} [/INST]
            """
            return self.llm.predict(prompt)
 
        # Function to combine and re-process if possible
        def combine_and_reprocess(chunks: List[str]) -> str:
            combined_text = " ".join(chunks)
            combined_token_count = len(self.tokenizer.encode(combined_text))
            if combined_token_count < 3900:
                return process_chunk(combined_text)
            else:
                # If still too large, split again and process separately
                mid_point = len(chunks) // 2
                first_half = combine_and_reprocess(chunks[:mid_point])
                second_half = combine_and_reprocess(chunks[mid_point:])
                return first_half + " " + second_half
 
        if token_count < 3900:
            return process_chunk(text)
        else:
            # Initial splitting of the text into manageable chunks
            initial_chunks = self.custom_text_splitter(text)
            # Process each chunk initially
            processed_chunks = [process_chunk(chunk) for chunk in initial_chunks]
            # Attempt to combine processed chunks and reprocess if under the limit
            return combine_and_reprocess(processed_chunks)
       
    def actions_taken_split(self, text: str, token_count: int) -> str:
        """
        Identify actions taken in the conversation, splitting if necessary.

        Args:
            text (str): The input text to analyze.
            token_count (int): The number of tokens in the input text.

        Returns:
            str: A description of actions taken.
        """
        # Helper function to process text chunks with the LLM
        def process_chunk(text_chunk: str) -> str:
            prompt = f"""
            [INST] <<SYS>> For each conversation, analyze the interaction between the customer and the agent. Identify and return a detailed list of the specific steps or actions taken by the agent to address and resolve the customer's query or issue in under 150 words.<<SYS>> {text_chunk} [/INST]
            """
            return self.llm.predict(prompt)
 
        # Function to combine and re-process if possible
        def combine_and_reprocess(chunks: List[str]) -> str:
            combined_text = " ".join(chunks)
            combined_token_count = len(self.tokenizer.encode(combined_text))
            if combined_token_count < 3900:
                return process_chunk(combined_text)
            else:
                # If still too large, split again and process separately
                mid_point = len(chunks) // 2
                first_half = combine_and_reprocess(chunks[:mid_point])
                second_half = combine_and_reprocess(chunks[mid_point:])
                return first_half + " " + second_half
 
        if token_count < 3900:
            return process_chunk(text)
        else:
            # Initial splitting of the text into manageable chunks
            initial_chunks = self.custom_text_splitter(text)
            # Process each chunk initially
            processed_chunks = [process_chunk(chunk) for chunk in initial_chunks]
            # Attempt to combine processed chunks and reprocess if under the limit
            return combine_and_reprocess(processed_chunks)

def load_credentials(filepath: str) -> Dict[str, str]:
    """
    Load credentials from a JSON file.

    Args:
        filepath (str): Path to the JSON file containing credentials.

    Returns:
        Dict[str, str]: A dictionary with 'input_path' and 'output_path' from the JSON file.
    """
    with open(filepath, "r") as f:
        creds = json.load(f)
    return creds
    
creds = load_credentials("/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/CallVoltMasterRepo/RepeatCallers/credentials/summary.json")
file_path = creds['input_path']
save_path = creds['output_path']

# Read the Parquet file into a DataFrame
try:
    front_end = pd.read_parquet(file_path)
    print(f"Successfully loaded DataFrame from: {file_path}")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"Error: Failed to read DataFrame from {file_path}. Exception: {e}")

 
start = time.time()

model_uri = setup_llama()
tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
summarizer = LlamaSummarizer(model_uri)

front_end['transcription'] = front_end['redacted'].apply(summarizer.merger)
front_end['word_count'] = front_end['transcription'].apply(summarizer.word_count)
front_end['token_count'] = front_end['transcription'].apply(lambda x: len(tokenizer.encode(x)))
front_end['summary'] = front_end.apply(lambda x: summarizer.summary(x['transcription'], x['token_count']) if x['word_count'] > 50 else "No summary generated", axis=1)
front_end['actions_taken'] = front_end.apply(lambda x: summarizer.actions_taken(x['transcription'], x['token_count']) if x['word_count'] > 50 else "No actions generated", axis=1)

print("Job completed in "+str(time.time()-start))

# Save the DataFrame to Parquet format
try:
    front_end.to_parquet(save_path, index=False)
    print(f"DataFrame successfully saved to: {save_path}")
except Exception as e:
    print(f"Error: Failed to save DataFrame to {save_path}. Exception: {e}")
