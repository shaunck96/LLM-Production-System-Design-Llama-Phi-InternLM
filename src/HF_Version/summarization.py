import os
import pandas as pd
import ray
import time
from datetime import datetime, timedelta
import logging
from huggingface_hub import hf_hub_download
from langchain.llms import LlamaCpp
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nltk.tokenize import word_tokenize
import nltk
from itertools import cycle
from transformers import AutoTokenizer, LlamaTokenizerFast

nltk.download('punkt')

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
MODEL_NAME_OR_PATH = "TheBloke/Llama-2-7B-Chat-GGUF"
MODEL_BASENAME = "llama-2-7b-chat.Q5_K_S.gguf"
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '10'))
N_GPU_LAYERS = int(os.getenv('N_GPU_LAYERS', '-1'))
N_BATCH = int(os.getenv('N_BATCH', '512'))

# Download model
def download_model(repo_id: str, filename: str) -> str:
    """
    Download the model from Hugging Face Hub.

    Args:
        repo_id (str): The repository ID of the model.
        filename (str): The filename of the model to download.

    Returns:
        str: The local path to the downloaded model.
    """
    return hf_hub_download(repo_id=repo_id, filename=filename)

model_path = download_model(MODEL_NAME_OR_PATH, MODEL_BASENAME)

# Initialize Ray
ray.init()

@ray.remote(num_gpus=1)
class TranscriptProcessor:
    def __init__(self, model_path: str) -> None:
        """
        Initialize the TranscriptProcessor with the given model path.

        Args:
            model_path (str): The path to the downloaded model.
        """
        self.llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=N_GPU_LAYERS,
            n_batch=N_BATCH,
            n_ctx=4096
        )
        self.tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer") #AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    def process_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a batch of transcripts by applying various transformations.

        Args:
            df (pd.DataFrame): DataFrame containing the transcripts to process.

        Returns:
            pd.DataFrame: DataFrame with processed transcripts including summaries and actions taken.
        """
        df['transcription'] = df['anonymized'].apply(self.merger)
        df['word_count'] = df['transcription'].apply(self.word_count)
        df['token_count'] = df['transcription'].apply(lambda x: len(self.tokenizer.encode(x)))
        # Apply summary and actions_taken only if word count > 50
        df['summary'] = df.apply(lambda x: self.summary(x['transcription'], x['token_count']) if x['word_count'] > 50 else "No summary generated", axis=1)
        df['actions_taken'] = df.apply(lambda x: self.actions_taken(x['transcription'], x['token_count']) if x['word_count'] > 50 else "No actions generated", axis=1)
        return df
    
    def custom_text_splitter(self, 
                             text: str, 
                             chunk_size: int = 3200, 
                             max_tokens: int = 3800, 
                             overlap: Optional[int] = None) -> list[str]:
        """
        Split the text into chunks based on the specified chunk size and token limit.

        Args:
            text (str): The text to split.
            chunk_size (int): The size of each chunk in characters.
            max_tokens (int): The maximum number of tokens per chunk.
            overlap (Optional[int]): The number of overlapping characters between chunks.

        Returns:
            list[str]: List of text chunks.
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
    
    def merger(self, text: str) -> str:
        """
        Process and clean the text for merging.

        Args:
            text (str): The text to process.

        Returns:
            str: The processed text.
        """
        return " ".join([trans['text'] for trans in anonymized_list])

    def word_count(self, text: str) -> int:
        """
        Count the number of words in the text.

        Args:
            text (str): The text to count words.

        Returns:
            int: The word count.
        """
        return len(word_tokenize(text))

    def summary(self, text: str, token_count: int) -> str:
        """
        Generate a summary of the text if the word count is greater than 50.

        Args:
            text (str): The text to summarize.
            token_count (int): The number of tokens in the text.

        Returns:
            str: The summary of the text.
        """
        prompt = """
    [INST] <<SYS>> Summarize the key discussion points between the customer and agent in under 150 words.
    <<SYS>> {} [/INST]
    """
        if token_count>3900:
            chunks = self.custom_text_splitter(text)
            chunk_summaries = []
            for chunk in chunks:
                summary = self.llm.invoke(prompt.format(chunk))
                chunk_summaries.append(summary)
            combined_summary = "\n".join(chunk_summaries)
            return combined_summary
        else:
            return(self.llm.invoke(prompt.format(text)))

    def actions_taken(self, text: str, token_count: int) -> str:
        """
        Determine actions taken based on the text content.

        Args:
            text (str): The text to evaluate.
            token_count (int): The number of tokens in the text.

        Returns:
            str: Actions taken based on the text.
        """
        if token_count<3900:
            prompt = f"[INST] <<SYS>> For each conversation, analyze the interaction between the customer and the agent. Identify and return a detailed list of the specific steps or actions taken by the agent to address and resolve the customer's query or issue in under 150 words.<<SYS>> {text} [/INST]"
            return self.llm.invoke(prompt)
        else:
            return "No actions generated"

    def summary_split(self, text: str, token_count: int) -> str:
        """
        Splits text into manageable chunks and generates a summary for each chunk using the LLM.

        Args:
            text (str): The text to summarize.
            token_count (int): The number of tokens in the text.

        Returns:
            str: The generated summary.
        """
        # Helper function to process text chunks with the LLM
        def process_chunk(text_chunk):
            prompt = f"""
            [INST] <<SYS>> Summarize the key discussion points between the customer and agent in under 150 words.
            <<SYS>> {text_chunk} [/INST]
            """
            return self.llm.invoke(prompt)

        # Function to combine and re-process if possible
        def combine_and_reprocess(chunks):
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
        Splits text into manageable chunks and analyzes actions taken by the agent using the LLM.

        Args:
            text (str): The text to analyze.
            token_count (int): The number of tokens in the text.

        Returns:
            str: The detailed list of actions taken.
        """
        # Helper function to process text chunks with the LLM
        def process_chunk(text_chunk):
            prompt = f"""
            [INST] <<SYS>> For each conversation, analyze the interaction between the customer and the agent. Identify and return a detailed list of the specific steps or actions taken by the agent to address and resolve the customer's query or issue in under 150 words.<<SYS>> {text_chunk} [/INST]
            """
            return self.llm.invoke(prompt)

        # Function to combine and re-process if possible
        def combine_and_reprocess(chunks):
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
        
# Create actors
num_actors = 2
processors = [TranscriptProcessor.remote(model_path) for _ in range(num_actors)]

def process_file(file_path: str, processors: List[ray.remote(TranscriptProcessor)]) -> pd.DataFrame:
    """
    Processes a parquet file by loading it into a DataFrame, splitting it into batches,
    and processing each batch using distributed processors.

    Args:
        file_path (str): The path to the parquet file to be processed.
        processors (List[ray.remote(TranscriptProcessor)]): A list of distributed TranscriptProcessor instances.

    Returns:
        pd.DataFrame: The concatenated results of all processed batches.
    """
    try:
        df = pd.read_parquet(file_path)
        df = df.iloc[:10, :]
        batches = [df.iloc[i:i + BATCH_SIZE] for i in range(0, len(df), BATCH_SIZE)]
        futures = [processor.process_batch.remote(batch) for processor, batch in zip(cycle(processors), batches)]
        results = ray.get(futures)
        return pd.concat(results)
    except Exception as e:
        logging.error(f"Failed to process file {file_path}: {e}")
        return pd.DataFrame()

with open("/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/CallVoltMasterRepo/RepeatCallers/credentials/summary.json", "r") as f:
    creds = json.load(f)

today = datetime.now().date() - timedelta(days=2)
today_str = today.strftime('%Y_%m_%d')
file_path = creds['input_path']
save_path = creds['output_path']

# Process the data
start_time = time.time()
processed_df = process_file(file_path)

# Save the processed DataFrame
try:
    print(processed_df)
    processed_df.to_parquet(save_path)
    logging.info(f"DataFrame successfully saved to: {save_path}")
except Exception as e:
    logging.error(f"Error: Failed to save DataFrame to {save_path}. Exception: {e}")

ray.shutdown()
logging.info(f"Job completed in {time.time() - start_time} seconds")
