import torch
import pandas as pd
import time
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import json
import numpy as np
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import json
import regex as re
from nltk.tokenize import sent_tokenize
import ast
import time
import json
import ast
import ray
from datetime import datetime, timedelta
import os

class TopicModelling:
    def __init__(self, device: int):
        """
        Initialize the TopicModelling class.

        Args:
            device (int): The GPU device number to use.
        """
        self.device = device
        torch.cuda.set_device(self.device)  # Explicitly setting device
        self.setup()

    def setup(self) -> None:
        """
        Set up the necessary models and load topic data.
        """
        self.classifier = pipeline("zero-shot-classification", 
                                    model="facebook/bart-large-mnli",
                                    device=self.device)
        
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5',
                                          device=self.device).to(self.device)
        
        with open(r"/Workspace/Users/sshibu@pplweb.com/CS_PA_NLP/mp3_files/topics/topic_og.json", "r") as f:
            json_data = f.read()

            data = json.loads(json_data)
            
        self.ci = data
        self.topic_embeddings = self.model.encode(
            (list(self.ci.values())))
        
    def probability_assignment(self, summary: str, topic_list: List[str]) -> Union[Dict[str, List[Union[str, float]]], str]:
        """
        Assign probabilities to topics for a given summary.

        Args:
            summary (str): The text summary to classify.
            topic_list (List[str]): List of potential topics.

        Returns:
            Union[Dict[str, List[Union[str, float]]], str]: Classification results or error message.
        """
        try:
            if len(topic_list) == 0:
                return "UNIDENTIFIED"
            return self.classifier(summary, topic_list)
        except Exception as e:
            print(f"Error in probability_assignment: {str(e)}")
            return "ERROR"

    def apply_probability_assignment(self, topic_list: List[str], summary: str) -> Union[Dict[str, List[Union[str, float]]], str]:
        """
        Apply probability assignment to a summary given a list of topics.

        Args:
            topic_list (List[str]): List of potential topics.
            summary (str): The text summary to classify.

        Returns:
            Union[Dict[str, List[Union[str, float]]], str]: Classification results or error message.
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

    def parse_topic_with_probabilities(self, x: Union[Dict, Any]) -> Dict[str, float]:
        """
        Parse topic probabilities from classification results.

        Args:
            x (Union[Dict, Any]): Classification results or any other input.

        Returns:
            Dict[str, float]: Parsed topic probabilities or default dictionary.
        """
        try:
            if type(x) is dict:
                return x
        except (IndexError, ValueError, SyntaxError):
            pass
        return {'Unidentified': 1}

    def get_primary_topic(self, x: Dict[str, List[Union[str, float]]]) -> str:
        """
        Get the primary topic from classification results.

        Args:
            x (Dict[str, List[Union[str, float]]]): Classification results.

        Returns:
            str: Primary topic or 'Unidentified'.
        """
        try:
            return x[list(x.keys())[1]][0]
        except (IndexError, TypeError):
            return 'Unidentified'

    def get_secondary_topic(self, x: Dict[str, List[Union[str, float]]]) -> str:
        """
        Get the secondary topic from classification results.

        Args:
            x (Dict[str, List[Union[str, float]]]): Classification results.

        Returns:
            str: Secondary topic, 'None', or 'Unidentified'.
        """
        try:
            if len(list(x.keys())) > 1:
                return x[list(x.keys())[1]][1]
            else:
                return 'None'
        except (IndexError, TypeError):
            return 'None'
        
    def predict(self, summary: str) -> List[str]:
        """
        Predict primary and secondary topics for a given summary.

        Args:
            summary (str): The text summary to classify.

        Returns:
            List[str]: A list containing primary and secondary topics.
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

def safe_literal_eval(s: Union[str, List]) -> Optional[Union[List, Dict]]:
    """
    Safely evaluate a string containing a Python literal or a JSON object.

    Args:
        s (Union[str, List]): The string to evaluate or a list.

    Returns:
        Optional[Union[List, Dict]]: The evaluated object or None if parsing fails.
    """
    if isinstance(s, list):
        return s

    if isinstance(s, str):
        try:
            return ast.literal_eval(s)
        except ValueError as e:
            print(f"Error parsing string with ast.literal_eval: {e}\nAttempting to parse with json.loads.")
            try:
                return json.loads(s.replace("'", '"'))
            except json.JSONDecodeError as je:
                print(f"Error parsing string with json: {je}\nInput data: {s}")

    print(f"Unsupported data type or malformed input: {type(s)}\nInput data: {s}")
    return None

@ray.remote(num_gpus=1)
def process_texts_on_gpu(device_id: int, texts: List[str]) -> List[Dict[str, Union[str, List[str], float]]]:
    """
    Process a batch of texts on a GPU using the TopicModelling class.

    Args:
        device_id (int): The GPU device number to use.
        texts (List[str]): A list of text summaries to process.

    Returns:
        List[Dict[str, Union[str, List[str], float]]]: A list of dictionaries containing processed results.
    """
    tm = TopicModelling(device=0)
    results = []
    for text in texts:
        start_time = time.time()
        topics = tm.predict(text)
        end_time = time.time()
        # Ensure each entry is a dictionary with the required keys
        result_dict = {
            'summary': text,
            'topics': topics,  # Ensure this is a list or tuple of primary, secondary topics
            'time_taken': end_time - start_time
        }
        results.append(result_dict)
    return results

def main() -> None:
    """
    Main function to orchestrate the topic modelling process using Ray for distributed computing.
    """
    with open("/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/CallVoltMasterRepo/RepeatCallers/credentials/topics.json", "r") as f:
        creds = json.load(f)

    ray.init(num_gpus=2)

    today = datetime.now().date() - timedelta(days=2)
    today_str = today.strftime('%Y_%m_%d')
    file_path = creds['input_path']

    try:
        transcriptions = pd.read_parquet(file_path)
        print(f"Successfully loaded DataFrame from: {file_path}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except Exception as e:
        print(f"Error: Failed to read DataFrame from {file_path}. Exception: {e}")
        return

    file_names = transcriptions['summary'].tolist()
    start = time.time()
    print("Process Started")

    batch_size = len(file_names) // 2  # Splitting into two batches for two GPUs
    batches = [file_names[i * batch_size:(i + 1) * batch_size] for i in range(2)]

    results = []

    # Start processing texts on GPUs
    futures = [process_texts_on_gpu.remote(i, batches[i]) for i in range(2)]
    results.extend(ray.get(futures))

    final_results = []
    for batch_result in results:
        for item in batch_result:
            final_results.append({
                'summary': item['summary'],
                'topics': item['topics'],
                'time_taken': item['time_taken']
            })

    # Create the final DataFrame
    final_df = pd.DataFrame(final_results)

    save_path = creds['output_path']
    try:
        final_df.to_parquet(save_path)
        print(f"DataFrame successfully saved to: {save_path}")
    except Exception as e:
        print(f"Error: Failed to save DataFrame. Exception: {e}")

    ray.shutdown()
    print("Ray shutdown complete.", f"Time taken: {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    main()
