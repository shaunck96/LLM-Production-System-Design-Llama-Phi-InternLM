import mlflow.pyfunc
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd
import re
from nltk.tokenize import sent_tokenize
from datetime import datetime, timedelta
import ast
from collections import Counter
import time
from concurrent.futures import ThreadPoolExecutor
import os
from typing import Dict, List, Any, Optional

class Sentiment_0(mlflow.pyfunc.PythonModel):
    """Base class for sentiment analysis models."""

    def load_context(self, context):
        """
        Load the sentiment analysis model.

        Args:
            context: The MLflow context.
        """
        # Load the tokenizer and model for Summarization
        self.sent_path = "cardiffnlp/twitter-roberta-base-sentiment"
        self.pipe = pipeline("text-classification", model=self.sent_path, device=0) 

    def segment_sentiment_computer(self, chunk: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Calculate sentiment labels for segments within a chunk.

        Args:
            chunk (list): List of segments, each containing 'text' field.

        Returns:
            list: List of segments with 'sentiment' field added,
            indicating the sentiment label
            ('Positive', 'Negative', or 'Neutral')
            for each segment.

        Iterates through the segments in the chunk and calculates
        sentiment labels for each segment.
        """
        for segment in chunk:
            segment['sentiment'] = self.sentiment_computer(segment['text'])
        return chunk    
    
    def predict(self, context, input_text):
        sentiment = self.segment_sentiment_computer(input_text) if len(input_text) > 0 else "No Sentiment Identified"
        return sentiment
    
    def predict(self, context, input_text: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Calculate the most frequent sentiment label for a chunk of text.

        Args:
            chunk (str): Input text chunk.

        Returns:
            str: Most frequent sentiment label
            ('Positive', 'Negative', or 'Neutral') in the chunk.

        Splits the chunk into sentences and
        calculates the sentiment for each sentence,
        then determines the most frequent sentiment
        label in the chunk.
        """
        tokens_sent = re.compile('[.!?] ').split(chunk)
        sentiment_list = []
        for sentence in tokens_sent:
            sentiment_list.append(self.sentiment_scores(sentence))
        counts = Counter(sentiment_list)
        most_frequent_sentiment = counts.most_common(1)[0][0]
        return (most_frequent_sentiment)
    
    def sentiment_computer(self, chunk: str) -> str:
        """
        Calculate sentiment score for a given sentence.

        Args:
            sentence (str): Input sentence.

        Returns:
            str: Sentiment label ('Positive', 'Negative', or 'Neutral').

        Uses the SentimentIntensityAnalyzer from the nltk library
        to calculate the sentiment score and classify
        it as Positive, Negative, or Neutral.
        """

        return (self.pipe(sentence)[0]['label'])


class Sentiment_1(mlflow.pyfunc.PythonModel):
    """Base class for sentiment analysis models."""

    def load_context(self, context):
        """
        Load the sentiment analysis model.

        Args:
            context: The MLflow context.
        """
        self.sent_path = "cardiffnlp/twitter-roberta-base-sentiment"
        self.pipe = pipeline("text-classification", model=self.sent_path, device=1) 

    def segment_sentiment_computer(self, chunk: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Calculate sentiment labels for segments within a chunk.

        Args:
            chunk (list): List of segments, each containing 'text' field.

        Returns:
            list: List of segments with 'sentiment' field added,
            indicating the sentiment label
            ('Positive', 'Negative', or 'Neutral')
            for each segment.

        Iterates through the segments in the chunk and calculates
        sentiment labels for each segment.
        """
        for segment in chunk:
            segment['sentiment'] = self.sentiment_computer(segment['text'])
        return chunk    
    
    def predict(self, context, input_text: List[Dict[str, str]]) -> List[Dict[str, str]]:
        sentiment = self.segment_sentiment_computer(input_text) if len(input_text) > 0 else "No Sentiment Identified"
        return sentiment
    
    def sentiment_computer(self, chunk: str) -> str:
        """
        Calculate the most frequent sentiment label for a chunk of text.

        Args:
            chunk (str): Input text chunk.

        Returns:
            str: Most frequent sentiment label
            ('Positive', 'Negative', or 'Neutral') in the chunk.

        Splits the chunk into sentences and
        calculates the sentiment for each sentence,
        then determines the most frequent sentiment
        label in the chunk.
        """
        tokens_sent = re.compile('[.!?] ').split(chunk)
        sentiment_list = []
        for sentence in tokens_sent:
            sentiment_list.append(self.sentiment_scores(sentence))
        counts = Counter(sentiment_list)
        most_frequent_sentiment = counts.most_common(1)[0][0]
        return (most_frequent_sentiment)
    
    def sentiment_scores(self, sentence: str) -> str:
        """
        Calculate sentiment score for a given sentence.

        Args:
            sentence (str): Input sentence.

        Returns:
            str: Sentiment label ('Positive', 'Negative', or 'Neutral').

        Uses the SentimentIntensityAnalyzer from the nltk library
        to calculate the sentiment score and classify
        it as Positive, Negative, or Neutral.
        """

        return (self.pipe(sentence)[0]['label'])

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

def load_models_on_gpu() -> tuple:
    """
    Load sentiment analysis models on GPUs.

    Returns:
        Tuple of loaded models for GPU 0 and GPU 1.
    """
    os.makedirs('sentiment_pipeline1', exist_ok=True)
    os.makedirs('sentiment_pipeline2', exist_ok=True)

    with mlflow.start_run() as run0:
        mlflow.pyfunc.log_model(
            artifact_path="sentiment_pipeline1",  # Specify the relative path within the run's artifacts
            python_model=Sentiment_0(),
            artifacts={"sentiment_pipeline1": "./sentiment_pipeline1"}  # Adjust path to correct location
        )


    model_uri0 = f"runs:/{run0.info.run_id}/sentiment_pipeline1"  # Construct model URI correctly

    model_0 = mlflow.pyfunc.load_model(model_uri0)

    with mlflow.start_run() as run1:
        mlflow.pyfunc.log_model(
            artifact_path="sentiment_pipeline2",  # Specify the relative path within the run's artifacts
            python_model=Sentiment_1(),
            artifacts={"sentiment_pipeline2": "./sentiment_pipeline2"}  # Adjust path to correct location
        )

    model_uri1 = f"runs:/{run1.info.run_id}/sentiment_pipeline2"  # Construct model URI correctly

    model_1 = mlflow.pyfunc.load_model(model_uri1)

    return model_0, model_1

def sa_files_with_model(model_instance: mlflow.pyfunc.PyFuncModel, files: List[str]) -> List[Dict[str, str]]:
    """
    Perform sentiment analysis on files using the given model.

    Args:
        model_instance: Loaded sentiment analysis model.
        files: List of file contents to analyze.

    Returns:
        List of dictionaries containing original text and sentiment.
    """
    sa = model_instance
    results = []
    for transcription in files:
        start_time = time.time()
        sentiment = sa.predict(transcription)
        end_time = time.time()
        results.append({
            'original':transcription,
            'sentiment':sentiment,
        })
    return results

model_0, model_1 = load_models_on_gpu()
creds = load_credentials("/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/CallVoltMasterRepo/RepeatCallers/credentials/sentiment.json")
transcriptions = pd.read_parquet(creds['input_path'])
file_names = transcriptions['transcription'].tolist()

half_size = len(file_names) // 2
files_gpu_0 = file_names[:half_size]
files_gpu_1 = file_names[half_size:]
results_gpu0 = []
results_gpu1 = []

with ThreadPoolExecutor(max_workers=2) as executor:
    # Submit tasks for each GPU
    future_gpu0 = executor.submit(sa_files_with_model, model_0, files_gpu_0)
    future_gpu1 = executor.submit(sa_files_with_model, model_1, files_gpu_1)

    # Get results from futures
    results_gpu0 = future_gpu0.result()
    results_gpu1 = future_gpu1.result()

df_gpu0 = pd.DataFrame(results_gpu0)
df_gpu1 = pd.DataFrame(results_gpu1)
final_df = pd.concat([df_gpu0, df_gpu1], ignore_index=True)

try:
    final_df.to_parquet(save_path, index=False)
    print(f"DataFrame successfully saved to: {creds['output_path']}")
except Exception as e:
    print(f"Error: Failed to save DataFrame to {creds['output_path']}. {str(e)}")
