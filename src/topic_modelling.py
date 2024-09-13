import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import regex as re
from nltk.tokenize import sent_tokenize
import time
import json
import ast
from datetime import datetime, timedelta
import os
import mlflow
from concurrent.futures import ThreadPoolExecutor

class TopicModelling_0(mlflow.pyfunc.PythonModel):
    """
    A class for topic modeling using zero-shot classification and sentence embeddings.
    """
    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Load the necessary models and data for topic modeling.

        Args:
            context (mlflow.pyfunc.PythonModelContext): The MLflow context.
        """
        self.classifier = pipeline("zero-shot-classification", 
                                    model="facebook/bart-large-mnli",
                                    device=0)
        
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5',
                                          device=0).to(0)
        
        with open(r"/Workspace/Users/sshibu@pplweb.com/CS_PA_NLP/mp3_files/topics/topic_og.json", "r") as f:
            json_data = f.read()

            data = json.loads(json_data)
            
        self.ci = data
        self.topic_embeddings = self.model.encode(
            (list(self.ci.values())))
        
    def probability_assignment(self, summary: str, topic_list: List[str]) -> Dict[str, Any]:
        """
        Assign probabilities to topics for a given summary.

        Args:
            summary (str): The text summary to classify.
            topic_list (List[str]): List of potential topics.

        Returns:
            Dict[str, Any]: Classification results or "UNIDENTIFIED" if topic_list is empty.
        """
        try:
            if len(topic_list) == 0:
                return "UNIDENTIFIED"
            return self.classifier(summary, topic_list)
        except Exception as e:
            print(f"Error in probability_assignment: {str(e)}")
            return "ERROR"

    def apply_probability_assignment(self, topic_list: List[str], summary: str) -> Union[Dict[str, Any], str]:
        """
        Apply probability assignment to the given summary for the provided topics.

        This method handles the probability assignment process, including error cases.

        Args:
            topic_list (List[str]): A list of topics to consider for probability assignment.
            summary (str): The text summary to analyze.

        Returns:
            Union[Dict[str, Any], str]: 
                - A dictionary of probability assignments if successful.
                - "UNIDENTIFIED" if the topic list is empty.
                - "ERROR" if an exception occurs during processing.
        """
        try:
            if len(topic_list) == 0:
                return "UNIDENTIFIED"
            else:
                probabilities = self.probability_assignment(
                    summary, topic_list)
                return probabilities
        except Exception as e:
            print(f"Error in apply_probability_assignment: {str(e)}")
            return "ERROR"

    def parse_topic_with_probabilities(self, x: Any) -> Dict[str, float]:
        """
        Parse the topic probabilities.

        Args:
            x (Any): The input to parse.

        Returns:
            Dict[str, float]: Parsed topic probabilities or {'Unidentified': 1} if parsing fails.
        """
        try:
            if type(x) is dict:
                return x
        except (IndexError, ValueError, SyntaxError):
            pass
        return {'Unidentified': 1}

    def get_primary_topic(self, x: Dict[str, Any]) -> str:
        """
        Get the primary topic from the classification results.

        Args:
            x (Dict[str, Any]): The classification results.

        Returns:
            str: The primary topic or 'Unidentified' if not found.
        """
        try:
            return x[list(x.keys())[1]][0]
        except (IndexError, TypeError):
            return 'Unidentified'

    def get_secondary_topic(self, x: Dict[str, Any]) -> str:
        """
        Get the secondary topic from the classification results.

        Args:
            x (Dict[str, Any]): The classification results.

        Returns:
            str: The secondary topic, 'None' if not found, or 'None' if there's only one topic.
        """
        try:
            if len(list(x.keys())) > 1:
                return x[list(x.keys())[1]][1]
            else:
                return 'None'
        except (IndexError, TypeError):
            return 'None'
        
    def predict(self, context: mlflow.pyfunc.PythonModelContext, summary: str) -> List[str]:
        """
        Predict topics for a given summary.

        Args:
            context (mlflow.pyfunc.PythonModelContext): The MLflow context.
            summary (str): The text summary to classify.

        Returns:
            List[str]: A list containing the primary and secondary topics.
        """
        try:
            index = 0
            threshold = 0.4
            top_2_topics_per_cluster = pd.DataFrame(
                columns=[
                    'Sentence',
                    'Topic',
                    'Position',
                    'cos_sim',
                    'Chunking Strategy'])
            
            chunks = list(summary.split('.'))
            chunks = [sentence for sentence in summary.split(
                '.') if len(sentence.split()) >= 5]
            
            sentence_embeddings = self.model.encode(chunks)
            
            for i, sentence_embedding in enumerate(sentence_embeddings):
                for topic_num, topic_embedding in enumerate(self.topic_embeddings):
                    dot_product = np.dot(sentence_embedding, 
                                         topic_embedding)
                    norm_A = np.linalg.norm(sentence_embedding)
                    norm_B = np.linalg.norm(topic_embedding)
                    cosine_similarity = dot_product / (norm_A * norm_B)
                    if cosine_similarity > threshold:
                        top_2_topics_per_cluster.at[index,
                                                    'Sentence'] = str(
                                                        chunks[i])
                        top_2_topics_per_cluster.at[index,
                                                    'Topic'] = str(
                                                        list(
                                                            self.ci.keys())[
                                                                topic_num])
                        top_2_topics_per_cluster.at[index,
                                                    'Position'] = i
                        top_2_topics_per_cluster.at[index,
                                                    'cos_sim'] = float(
                                                        cosine_similarity)
                        top_2_topics_per_cluster.at[index,
                                                    'Chunking Strategy'] = str(
                                                        chunks)
                        index += 1

            if len(top_2_topics_per_cluster) == 0:
                print("Empty top topics df")

            position_wise = top_2_topics_per_cluster.sort_values(by=[
                'Position'], ascending=True)
            if len(position_wise) >= 10:
                top_topics = list(position_wise.sort_values(by=[
                    'cos_sim'], ascending=False)['Topic'].iloc[0:10])
            elif len(position_wise) > 0:
                top_topics = list(position_wise.sort_values(by=[
                    'cos_sim'], ascending=False)['Topic'])
            else:
                top_topics = []

        except Exception as e:
            print(f"Error in topic_modeller: {str(e)}")
            return []

        topic_dict = self.parse_topic_with_probabilities(
            self.apply_probability_assignment(
                topic_list = top_topics, 
                summary = summary))
        primary = self.get_primary_topic(x = topic_dict)
        secondary = self.get_secondary_topic(x =topic_dict)

        return [primary, secondary]

class TopicModelling_1(mlflow.pyfunc.PythonModel):
    """
    A class for topic modeling using zero-shot classification and sentence embeddings.
    """
    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Load the necessary models and data for topic modeling.

        Args:
            context (mlflow.pyfunc.PythonModelContext): The MLflow context.
        """
        self.classifier = pipeline("zero-shot-classification", 
                                    model="facebook/bart-large-mnli",
                                    device=1)
        
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5',
                                          device=1).to(1)
        
        with open(r"/Workspace/Users/sshibu@pplweb.com/CS_PA_NLP/mp3_files/topics/topic_og.json", "r") as f:
            json_data = f.read()

            data = json.loads(json_data)
            
        self.ci = data
        self.topic_embeddings = self.model.encode(
            (list(self.ci.values())))
        
    def probability_assignment(self, summary: str, topic_list: List[str]) -> Dict[str, Any]:
        """
        Assign probabilities to topics for a given summary.

        Args:
            summary (str): The text summary to classify.
            topic_list (List[str]): List of potential topics.

        Returns:
            Dict[str, Any]: Classification results or "UNIDENTIFIED" if topic_list is empty.
        """
        try:
            if len(topic_list) == 0:
                return "UNIDENTIFIED"
            return self.classifier(summary, topic_list)
        except Exception as e:
            print(f"Error in probability_assignment: {str(e)}")
            return "ERROR"

    def apply_probability_assignment(self, topic_list: List[str], summary: str) -> Union[Dict[str, Any], str]:
        """
        Apply probability assignment to the given summary for the provided topics.

        This method handles the probability assignment process, including error cases.

        Args:
            topic_list (List[str]): A list of topics to consider for probability assignment.
            summary (str): The text summary to analyze.

        Returns:
            Union[Dict[str, Any], str]: 
                - A dictionary of probability assignments if successful.
                - "UNIDENTIFIED" if the topic list is empty.
                - "ERROR" if an exception occurs during processing.
        """
        try:
            if len(topic_list) == 0:
                return "UNIDENTIFIED"
            else:
                probabilities = self.probability_assignment(
                    summary, topic_list)
                return probabilities
        except Exception as e:
            print(f"Error in apply_probability_assignment: {str(e)}")
            return "ERROR"

    def parse_topic_with_probabilities(self, x: Any) -> Dict[str, float]:
        """
        Parse the topic probabilities.

        Args:
            x (Any): The input to parse.

        Returns:
            Dict[str, float]: Parsed topic probabilities or {'Unidentified': 1} if parsing fails.
        """
        try:
            if type(x) is dict:
                return x
        except (IndexError, ValueError, SyntaxError):
            pass
        return {'Unidentified': 1}

    def get_primary_topic(self, x: Dict[str, Any]) -> str:
        """
        Get the primary topic from the classification results.

        Args:
            x (Dict[str, Any]): The classification results.

        Returns:
            str: The primary topic or 'Unidentified' if not found.
        """
        try:
            return x[list(x.keys())[1]][0]
        except (IndexError, TypeError):
            return 'Unidentified'

    def get_secondary_topic(self, x: Dict[str, Any]) -> str:
        """
        Get the secondary topic from the classification results.

        Args:
            x (Dict[str, Any]): The classification results.

        Returns:
            str: The secondary topic, 'None' if not found, or 'None' if there's only one topic.
        """
        try:
            if len(list(x.keys())) > 1:
                return x[list(x.keys())[1]][1]
            else:
                return 'None'
        except (IndexError, TypeError):
            return 'None'
        
    def predict(self, context: mlflow.pyfunc.PythonModelContext, summary: str) -> List[str]:
        """
        Predict topics for a given summary.

        Args:
            context (mlflow.pyfunc.PythonModelContext): The MLflow context.
            summary (str): The text summary to classify.

        Returns:
            List[str]: A list containing the primary and secondary topics.
        """
        try:
            index = 0
            threshold = 0.4
            top_2_topics_per_cluster = pd.DataFrame(
                columns=[
                    'Sentence',
                    'Topic',
                    'Position',
                    'cos_sim',
                    'Chunking Strategy'])
            
            chunks = list(summary.split('.'))
            chunks = [sentence for sentence in summary.split(
                '.') if len(sentence.split()) >= 5]
            
            sentence_embeddings = self.model.encode(chunks)
            
            for i, sentence_embedding in enumerate(sentence_embeddings):
                for topic_num, topic_embedding in enumerate(self.topic_embeddings):
                    dot_product = np.dot(sentence_embedding, 
                                         topic_embedding)
                    norm_A = np.linalg.norm(sentence_embedding)
                    norm_B = np.linalg.norm(topic_embedding)
                    cosine_similarity = dot_product / (norm_A * norm_B)
                    if cosine_similarity > threshold:
                        top_2_topics_per_cluster.at[index,
                                                    'Sentence'] = str(
                                                        chunks[i])
                        top_2_topics_per_cluster.at[index,
                                                    'Topic'] = str(
                                                        list(
                                                            self.ci.keys())[
                                                                topic_num])
                        top_2_topics_per_cluster.at[index,
                                                    'Position'] = i
                        top_2_topics_per_cluster.at[index,
                                                    'cos_sim'] = float(
                                                        cosine_similarity)
                        top_2_topics_per_cluster.at[index,
                                                    'Chunking Strategy'] = str(
                                                        chunks)
                        index += 1

            if len(top_2_topics_per_cluster) == 0:
                print("Empty top topics df")

            position_wise = top_2_topics_per_cluster.sort_values(by=[
                'Position'], ascending=True)
            if len(position_wise) >= 10:
                top_topics = list(position_wise.sort_values(by=[
                    'cos_sim'], ascending=False)['Topic'].iloc[0:10])
            elif len(position_wise) > 0:
                top_topics = list(position_wise.sort_values(by=[
                    'cos_sim'], ascending=False)['Topic'])
            else:
                top_topics = []

        except Exception as e:
            print(f"Error in topic_modeller: {str(e)}")
            return []

        topic_dict = self.parse_topic_with_probabilities(
            self.apply_probability_assignment(
                topic_list = top_topics, 
                summary = summary))
        primary = self.get_primary_topic(x = topic_dict)
        secondary = self.get_secondary_topic(x =topic_dict)

        return [primary, secondary]


def tm_with_model(model_instance: TopicModelling, texts: List[str]) -> List[Dict[str, Any]]:
    """
    Process a list of texts using the given topic modeling instance.

    Args:
        model_instance (TopicModelling): An instance of the TopicModelling class.
        texts (List[str]): A list of text summaries to process.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the original text and predicted topics.
    """
    results = []
    for text in texts:
        try:
            topics = model_instance.predict(text)
            results.append({'text': text, 
                            'topics': topics})
        except Exception as e:
            print(f"Error processing text: {str(e)}")
            results.append({'text': text, 'topics': []})  # Append empty topics in case of error
    return results

def load_models_on_gpu() -> Tuple[mlflow.pyfunc.PyFuncModel, mlflow.pyfunc.PyFuncModel]:
    """
    Load topic modeling models on GPUs.

    Returns:
        Tuple[mlflow.pyfunc.PyFuncModel, mlflow.pyfunc.PyFuncModel]: Tuple containing two loaded models.
    """
    os.makedirs('tm_pipeline1', exist_ok=True)
    os.makedirs('tm_pipeline2', exist_ok=True)

    with mlflow.start_run() as run0:
        mlflow.pyfunc.log_model(
            artifact_path="tm_pipeline1",  # Specify the relative path within the run's artifacts
            python_model=TopicModelling_0(),
            artifacts={"tm_pipeline1": "./tm_pipeline1"}  # Adjust path to correct location
        )

    model_uri0 = f"runs:/{run0.info.run_id}/tm_pipeline1"  
    model_0 = mlflow.pyfunc.load_model(model_uri0)

    with mlflow.start_run() as run1:
        mlflow.pyfunc.log_model(
            artifact_path="tm_pipeline2",  # Specify the relative path within the run's artifacts
            python_model=TopicModelling_1(),
            artifacts={"tm_pipeline2": "./tm_pipeline2"}  # Adjust path to correct location
        )

    model_uri1 = f"runs:/{run1.info.run_id}/tm_pipeline2"
    model_1 = mlflow.pyfunc.load_model(model_uri1)

    return model_0, model_1

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

model_0, model_1 = load_models_on_gpu()
creds = load_credentials("/credentials/topics.json")

base_dir = "/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/Production_Code/transcriptions"
file_path = creds['input_path']
save_path = creds['output_path']

try:
    transcriptions = pd.read_parquet(file_path)
    print(f"Successfully loaded DataFrame from: {file_path}")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"Error: Failed to read DataFrame from {file_path}. Exception: {e}")

file_names = transcriptions['summary'].tolist()[:10]

start = time.time()
print("Process Started")

half_size = len(file_names) // 2
files_gpu_0 = file_names[:half_size]
files_gpu_1 = file_names[half_size:]
results_gpu0 = []
results_gpu1 = []

with ThreadPoolExecutor(max_workers=2) as executor:
    # Submit tasks for each GPU
    future_gpu0 = executor.submit(tm_with_model, model_0, files_gpu_0)
    future_gpu1 = executor.submit(tm_with_model, model_1, files_gpu_1)

    # Get results from futures
    results_gpu0 = future_gpu0.result()
    results_gpu1 = future_gpu1.result()

df_gpu0 = pd.DataFrame(results_gpu0)
df_gpu1 = pd.DataFrame(results_gpu1)
final_df = pd.concat([df_gpu0, df_gpu1], ignore_index=True)

try:
    final_df.to_parquet(save_path, index=False)
    print(f"DataFrame successfully saved to: {save_path}")
except Exception as e:
    print(f"Error: Failed to save DataFrame to {save_path}. {str(e)}")
