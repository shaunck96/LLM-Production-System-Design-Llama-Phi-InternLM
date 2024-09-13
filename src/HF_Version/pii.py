from presidio_analyzer import AnalyzerEngine, PatternRecognizer, RecognizerResult, EntityRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
import regex as re
import torch
import pandas as pd
import time
import os
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import ast
import json

class TextRedactor:
    """
    A class to perform text redaction by identifying and anonymizing sensitive information using
    various recognizers and pipelines.

    Attributes:
        device (int): The GPU device index to use.
        pipeline (transformers.Pipeline): The token classification pipeline for NER.
        analyzer (presidio_analyzer.AnalyzerEngine): The Presidio analyzer engine.
        anonymizer (presidio_anonymizer.AnonymizerEngine): The Presidio anonymizer engine.
        drm_path (str): Path to the de-identification model.
        drt (transformers.AutoTokenizer): The tokenizer for the de-identification model.
        drm (transformers.AutoModelForTokenClassification): The model for de-identification.
        redact_pii_pipeline (transformers.Pipeline): The NER pipeline for PII redaction.
    """
    def __init__(self, device: int):
        """
        Initializes the TextRedactor with the specified device index.

        Args:
            device (int): The GPU device index to use.
        """
        self.device = device
        torch.cuda.set_device(self.device)  # Explicitly setting device
        self.setup()

    def __init__(self, device: int):
        """
        Initializes the TextRedactor with the specified device index.

        Args:
            device (int): The GPU device index to use.
        """
        self.pipeline = pipeline(
            "token-classification",
            model="Jean-Baptiste/roberta-large-ner-english",
            aggregation_strategy="simple",
            device=self.device
        )
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.initialize_recognizers()
        self.drm_path = "StanfordAIMI/stanford-deidentifier-base"
        self.drt = AutoTokenizer.from_pretrained(self.drm_path)
        self.drm = AutoModelForTokenClassification.from_pretrained(self.drm_path)
        self.redact_pii_pipeline = pipeline("ner",
                                            model=self.drm,
                                            tokenizer=self.drt,
                                            aggregation_strategy='average',
                                            device=self.device)

    def load_config(self, file_path: str) -> Dict[str, Any]:
        """
        Loads the configuration from a JSON file.

        Args:
            file_path (str): The path to the JSON configuration file.

        Returns:
            Dict[str, Any]: The configuration data loaded from the file.
        """
        with open(file_path, 'r') as file:
            return json.load(file)

    def load_config(self, file_path: str) -> Dict[str, Any]:
        """
        Loads the configuration from a JSON file.

        Args:
            file_path (str): The path to the JSON configuration file.

        Returns:
            Dict[str, Any]: The configuration data loaded from the file.
        """
        config = self.load_config('/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/CallVoltMasterRepo/RepeatCallers/credentials/recognizers.json')

        # Titles recognizer
        titles_patterns = config['title_patterns']
        titles_recognizer = PatternRecognizer(supported_entity="TITLE", deny_list=titles_patterns)
        self.analyzer.registry.add_recognizer(titles_recognizer)

        # Transformers recognizer for named entity recognition
        transformers_config = config['transformers_recognizer']
        transformers_recognizer = self.HFTransformersRecognizer(self.pipeline, **transformers_config)
        self.analyzer.registry.add_recognizer(transformers_recognizer)

        # Additional pattern recognizers
        for pattern_key, pattern_config in config['pattern_recognizers'].items():
            if isinstance(pattern_config, list):  # handle cases where there are multiple patterns
                for pattern in pattern_config:
                    pat = Pattern(name=pattern['name'], regex=pattern['regex'], score=pattern['score'])
                    self.analyzer.registry.add_recognizer(PatternRecognizer(supported_entity=pattern_key.upper(), deny_list=[pat.regex]))
            else:
                pat = Pattern(name=pattern_config['name'], regex=pattern_config['regex'], score=pattern_config['score'])
                self.analyzer.registry.add_recognizer(PatternRecognizer(supported_entity=pattern_key.upper(), deny_list=[pat.regex]))

    def add_pattern_recognizer(self, entity_type: str, patterns: List[str], score: float) -> None:
        """
        Adds a pattern recognizer to the Presidio analyzer for the specified entity type.

        Args:
            entity_type (str): The type of entity to recognize.
            patterns (List[str]): List of regex patterns to recognize.
            score (float): The score associated with the patterns.
        """
        recognizer = PatternRecognizer(supported_entity=entity_type, patterns=[Pattern(name=f"{entity_type}_REGEX", regex=pattern, score=score) for pattern in patterns])
        self.analyzer.registry.add_recognizer(recognizer)

    class HFTransformersRecognizer(EntityRecognizer):
        """
        A custom entity recognizer using Hugging Face transformers for NER.

        Attributes:
            pipeline (transformers.Pipeline): The pipeline used for entity recognition.
        """
        def __init__(self, pipeline: pipeline, supported_entities: List[str], supported_language: str = "en"):
            """
            Initializes the HFTransformersRecognizer.

            Args:
                pipeline (transformers.Pipeline): The pipeline used for entity recognition.
                supported_entities (List[str]): List of supported entity types.
                supported_language (str): Language supported by the recognizer (default is "en").
            """
            super().__init__(supported_entities=supported_entities, supported_language=supported_language)
            self.pipeline = pipeline

        def analyze(self, text: str, entities: Optional[List[str]] = None, nlp_artifacts: Optional[Dict[str, Any]] = None) -> List[RecognizerResult]:
            """
            Analyzes the text for the supported entities using the pipeline.

            Args:
                text (str): The input text to analyze.
                entities (Optional[List[str]]): List of entity types to recognize (default is None, meaning all supported entities).
                nlp_artifacts (Optional[Dict[str, Any]]): Additional NLP artifacts (default is None).

            Returns:
                List[RecognizerResult]: List of results containing recognized entities.
            """
            results = []
            predictions = self.pipeline(text)
            for prediction in predictions:
                entity_type = prediction['entity_group']
                if entities is None or entity_type in entities:
                    results.append(RecognizerResult(entity_type=entity_type, start=prediction['start'], end=prediction['end'], score=prediction['score']))
            return results
        
    def add_pattern_recognizer(self, entity_type: str, patterns: List[str], score: float) -> None:
        """
        Adds a pattern recognizer to the Presidio analyzer for the specified entity type.

        Args:
            entity_type (str): The type of entity to recognize.
            patterns (List[str]): List of regex patterns to recognize.
            score (float): The score associated with the patterns.
        """
        pattern = r'\b\w+(-\w+)+\b'
        number_pattern = re.compile(r'\b\d+\b')
        pattern = r'\b\w+([.*\-_]+\w+)+\b'
        redacted_text = re.sub(pattern, '***', text)
        redacted_text = number_pattern.sub("[REDACTED]", redacted_text)
        redacted_text = re.sub(pattern, '[REDACTED]', redacted_text)
        pattern = r'\b(?:\d{4}[-\s]?){2,}(?=\d{4})'
        
        def repl(match: re.Match) -> str:
            account = match.group(0)
            # Replace all but the last 4 digits with 'X'
            return re.sub(r'\d', 'X', account[:-4]) + account[-4:]
        redacted_text = re.sub(pattern, repl, redacted_text)
        pattern = r'\b(?:(one|two|three|four|five|six|seven|eight|nine|zero)\s*(?:\s+and\s+)?)+\b'

        # Function to replace matched numbers with '[REDACTED]'
        def replace(match):
            return '[REDACTED]'

        # Use re.sub() to replace numbers in words with '[REDACTED]'
        redacted_text = re.sub(pattern, replace, redacted_text, flags=re.IGNORECASE)

        return redacted_text      

    def anonymize_text(self, transcription_list: List[Dict[str, Union[str, Any]]]) -> List[Dict[str, str]]:
        """
        Anonymizes text in the given list of transcriptions.

        Args:
            transcription_list (List[Dict[str, Union[str, Any]]]): List of transcriptions, each containing a 'text' key with the text to anonymize.

        Returns:
            List[Dict[str, str]]: List of transcriptions with anonymized text.
        """
        for transcription in transcription_list:
            results = self.analyzer.analyze(text=transcription['text'], language="en")
            anonymized_result = self.anonymizer.anonymize(text=transcription['text'], analyzer_results=results)
            anonymized_text = anonymized_result.text
            pii_words = [item['word'] for item in self.redact_pii_pipeline(anonymized_text)]
            for pii_word in pii_words:
                anonymized_text = re.sub(r'\b' + re.escape(pii_word) + r'\b', '*' * len(pii_word), anonymized_text)
            anonymized_text = self.redact_sequence_patterns(anonymized_text)
            transcription['text'] = anonymized_text
        return transcription_list

def process_texts(device_id: int, transcription_list: List[Dict[str, Union[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Processes the list of transcriptions by anonymizing text using a TextRedactor.

    Args:
        device_id (int): The GPU device index to use.
        transcription_list (List[Dict[str, Union[str, Any]]]): List of transcriptions to process.

    Returns:
        List[Dict[str, Any]]: List of results, each containing the anonymized text and time taken for processing.
    """
    redactor = TextRedactor(device=f'cuda:{device_id}')
    results = []
    for text in transcription_list:
        start_time = time.time()
        anonymized_text = redactor.anonymize_text(text)
        end_time = time.time()
        results.append({'anonymized': anonymized_text, 
                        'time_taken': end_time - start_time})
    return results

with open("/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/CallVoltMasterRepo/RepeatCallers/credentials/redact.json", "r") as f:
    creds = json.load(f)

transcriptions = pd.read_parquet(creds['input_path'])
file_names = transcriptions['transcription'].tolist()

num_gpus = 2
batch_size = len(file_names) // num_gpus
batches = [file_names[i * batch_size:(i + 1) * batch_size] for i in range(num_gpus)]

results = []
for i in range(num_gpus):
    device_results = process_texts(i, batches[i])
    results.extend(device_results)

for result in results:
    print(result)

pd.DataFrame(results).to_parquet(creds['output_path'])
