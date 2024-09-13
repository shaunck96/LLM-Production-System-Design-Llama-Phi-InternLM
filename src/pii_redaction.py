from presidio_analyzer import AnalyzerEngine, PatternRecognizer, RecognizerResult, EntityRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
import regex as re
import pandas as pd
import time
import os
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import mlflow
from concurrent.futures import ThreadPoolExecutor
import json
from typing import Dict, List, Any, Optional
from mlflow.pyfunc import PythonModel


class TextRedactorBase(PythonModel):
    """
    Base class for text redaction using various NLP models and techniques.

    This class provides functionality to redact sensitive information from text
    using a combination of named entity recognition, pattern matching, and
    custom recognizers.

    Attributes:
        device (int): The GPU device number to use for model inference.
        pipeline: HuggingFace pipeline for token classification.
        analyzer (AnalyzerEngine): Presidio analyzer for detecting PII.
        anonymizer (AnonymizerEngine): Presidio anonymizer for redacting PII.
        drm_path (str): Path to the Stanford deidentifier model.
        drt: AutoTokenizer for the Stanford deidentifier model.
        drm: AutoModelForTokenClassification for the Stanford deidentifier model.
        redact_pii_pipeline: HuggingFace pipeline for named entity recognition using the Stanford model.
    """
    def __init__(self, device: int):
        """
        Initialize the TextRedactorBase with a specified GPU device.

        Args:
            device (int): The GPU device number to use for model inference.
        """
        self.device = device

    def load_context(self, context: Dict[str, Any]) -> None:
        """
        Load and initialize all necessary components for text redaction.

        This method sets up the NLP pipelines, analyzers, and recognizers
        needed for identifying and redacting sensitive information in text.

        Args:
            context (Dict[str, Any]): A dictionary containing any additional
                context or configuration needed for initialization.

        Returns:
            None
        """
        self.pipeline = pipeline(
            "token-classification",
            model="Jean-Baptiste/roberta-large-ner-english",
            aggregation_strategy="simple",
            device=self.device
        )
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self._initialize_recognizers()
        self.drm_path = "StanfordAIMI/stanford-deidentifier-base"
        self.drt = AutoTokenizer.from_pretrained(self.drm_path)
        self.drm = AutoModelForTokenClassification.from_pretrained(self.drm_path)
        self.redact_pii_pipeline = pipeline("ner",
                                            model=self.drm,
                                            tokenizer=self.drt,
                                            aggregation_strategy='average',
                                            device=self.device)

    def _initialize_recognizers(self) -> None:
        """Initialize and add various recognizers to the analyzer."""
        # Titles recognizer
        titles_patterns = [r"\bMr\.\b", r"\bMrs\.\b", r"\bMiss\b"]
        titles_recognizer = PatternRecognizer(supported_entity="TITLE", deny_list=titles_patterns)
        self.analyzer.registry.add_recognizer(titles_recognizer)

        # Transformers recognizer for named entity recognition
        transformers_recognizer = self.HFTransformersRecognizer(self.pipeline, supported_entities=["PERSON", "LOCATION", "ORGANIZATION"])
        self.analyzer.registry.add_recognizer(transformers_recognizer)

        # Additional pattern recognizers
        phone_number_patterns = [Pattern(name="PHONE_NUMBER_REGEX",
                                            regex=r"\(?\b\d{3}\)?[-.]?\s?\d{3}[-.]?\s?\d{4}\b",  # noqa: E501
                                            score=0.5)]
        email_patterns = [Pattern(name="EMAIL_REGEX",
                                    regex=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # noqa: E501
                                    score=0.5)]
        account_number_patterns = [Pattern(name="ACCOUNT_NUMBER_REGEX",
                                            regex=r"\b\d{8,12}\b",
                                            score=0.5)]
        date_patterns = [Pattern(name="DATE_REGEX",
                                    regex=r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{2,4}[-/]\d{1,2}[-/]\d{1,2}|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b\s\d{1,2},?\s\d{4})\b",  # noqa: E501
                                    score=0.5)]
        address_patterns = [
            Pattern(name="US_ADDRESS_REGEX_1",
                    regex=r"\b\d{1,5}\s([a-zA-Z\s]{1,})\b,?\s([a-zA-Z\s]{1,}),?\s([A-Z]{2}),?\s\d{5}\b",  # noqa: E501
                    score=0.85),
        ]
        ssn_patterns = [
            Pattern(name="SSN_REGEX_FULL",
                    regex=r"\b\d{3}-\d{2}-\d{4}\b",  # noqa: E501
                    score=0.85),
            Pattern(name="SSN_REGEX_LAST4",
                    regex=r"\b\d{4}\b",
                    score=0.85)
        ]
        dollar_amount_patterns = [
            Pattern(name="DOLLAR_AMOUNT_REGEX",
                    regex=r"\$\s?\d+(?:,\d{3})*(?:\.\d{2})?",  # noqa: E501
                    score=0.6)

        ]
        bill_amount_patterns = [
            Pattern(name="BILL_AMOUNT_REGEX",
                    regex=r"\b(?:payment|bill|amount)\s?of\s?\$\s?\d+(?:,\d{3})*(?:\.\d{2})?",  # noqa: E501
                    score=0.6)
        ]
        confirmation_number_patterns = [
            Pattern(name="CONFIRMATION_NUMBER_REGEX",
                    regex=r"confirmation\snumber\s(?:is\s)?((?:\d+|\w+)(?:,\s?(?:\d+|\w+))*)",  # noqa: E501
                    score=0.9)
        ]
        passport_patterns = [Pattern(name="PASSPORT_REGEX", regex=r"\b[A-Z]{1,2}\d{7,8}\b", score=0.85)]
        drivers_license_patterns = [Pattern(name="DRIVERS_LICENSE_REGEX", regex=r"\b[A-Z]{2}\d{7,8}\b|\bDL\s?\d{8,12}\b", score=0.85)]
        tax_id_patterns = [Pattern(name="TAX_ID_REGEX", regex=r"\bITIN\s?\d{9}\b|\bEIN\s?\d{9}\b", score=0.85)]

        address_recognizer = PatternRecognizer(
            supported_entity="ADDRESS", patterns=address_patterns)
        ssn_recognizer = PatternRecognizer(
            supported_entity="US_SSN", patterns=ssn_patterns)
        phone_number_recognizer = PatternRecognizer(
            supported_entity="PHONE_NUMBER", patterns=phone_number_patterns)
        email_recognizer = PatternRecognizer(
            supported_entity="EMAIL_ADDRESS", patterns=email_patterns)
        account_number_recognizer = PatternRecognizer(
            supported_entity="ACCOUNT_NUMBER",
            patterns=account_number_patterns)
        date_recognizer = PatternRecognizer(
            supported_entity="DATE", patterns=date_patterns)
        dollar_amount_recognizer = PatternRecognizer(
            supported_entity="DOLLAR_AMOUNT", patterns=dollar_amount_patterns)
        bill_amount_recognizer = PatternRecognizer(
            supported_entity="BILL_AMOUNT", patterns=bill_amount_patterns)
        confirmation_number_recognizer = PatternRecognizer(
            supported_entity="CONFIRMATION_NUMBER",
            patterns=confirmation_number_patterns)
        passport_recognizer = PatternRecognizer(
            supported_entity="PASSPORT_REGEX",
            patterns=confirmation_number_patterns)
        drivers_license_recognizer = PatternRecognizer(
            supported_entity="DRIVERS_LICENSE_REGEX",
            patterns=confirmation_number_patterns)
        self.analyzer.registry.add_recognizer(phone_number_recognizer)
        self.analyzer.registry.add_recognizer(email_recognizer)
        self.analyzer.registry.add_recognizer(account_number_recognizer)
        self.analyzer.registry.add_recognizer(date_recognizer)
        self.analyzer.registry.add_recognizer(address_recognizer)
        self.analyzer.registry.add_recognizer(ssn_recognizer)
        self.analyzer.registry.add_recognizer(dollar_amount_recognizer)
        self.analyzer.registry.add_recognizer(bill_amount_recognizer)
        self.analyzer.registry.add_recognizer(confirmation_number_recognizer)
        self.analyzer.registry.add_recognizer(passport_recognizer)
        self.analyzer.registry.add_recognizer(drivers_license_recognizer)

    def _add_pattern_recognizer(self, entity_type: str, patterns: List[str], score: float) -> None:
        """
        Add a pattern recognizer to the analyzer.

        Args:
            entity_type (str): The type of entity to recognize.
            patterns (List[str]): List of regex patterns for the entity.
            score (float): Confidence score for the recognizer.
        """
        recognizer = PatternRecognizer(supported_entity=entity_type, patterns=[Pattern(name=f"{entity_type}_REGEX", regex=pattern, score=score) for pattern in patterns])
        self.analyzer.registry.add_recognizer(recognizer)

    class HFTransformersRecognizer(EntityRecognizer):
        """Custom recognizer using Hugging Face Transformers."""
        def __init__(self, pipeline: Any, supported_entities: List[str], supported_language: str = "en"):
            """
            Initialize the HFTransformersRecognizer.

            Args:
                pipeline (Any): The Hugging Face pipeline for token classification.
                supported_entities (List[str]): List of entity types supported by this recognizer.
                supported_language (str, optional): The language supported by this recognizer. Defaults to "en".
            """
            super().__init__(supported_entities=supported_entities, supported_language=supported_language)
            self.pipeline = pipeline

        def analyze(self, text: str, entities: Optional[List[str]] = None, nlp_artifacts: Optional[Any] = None) -> List[RecognizerResult]:
            """
            Analyze the text and return recognized entities.

            Args:
                text (str): The text to analyze.
                entities (Optional[List[str]], optional): List of entities to look for. Defaults to None.
                nlp_artifacts (Optional[Any], optional): Any NLP artifacts to use in analysis. Defaults to None.

            Returns:
                List[RecognizerResult]: List of recognized entities.
            """
            results = []
            predictions = self.pipeline(text)
            for prediction in predictions:
                entity_type = prediction['entity_group']
                if entities is None or entity_type in entities:
                    results.append(RecognizerResult(entity_type=entity_type, start=prediction['start'], end=prediction['end'], score=prediction['score']))
            return results
        
    def redact_sequence_patterns(self, text: str) -> str:
        """
        Redact various patterns in the text.

        Args:
            text (str): The text to redact.

        Returns:
            str: The redacted text.
        """
        pattern = r'\b\w+(-\w+)+\b'
        number_pattern = re.compile(r'\b\d+\b')
        pattern = r'\b\w+([.*\-_]+\w+)+\b'
        redacted_text = re.sub(pattern, '***', text)
        redacted_text = number_pattern.sub("[REDACTED]", redacted_text)
        redacted_text = re.sub(pattern, '[REDACTED]', redacted_text)
        pattern = r'\b(?:\d{4}[-\s]?){2,}(?=\d{4})'
        
        def repl(match):
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

    def predict(self, context: Any, transcription_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict and redact sensitive information from a list of transcriptions.

        Args:
            context (Any): The context for prediction (unused in this method).
            transcription_list (List[Dict[str, Any]]): List of transcriptions to process.

        Returns:
            List[Dict[str, Any]]: List of processed transcriptions with redacted text.
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


class TextRedactor_0(TextRedactorBase):
    """
    A specific implementation of TextRedactorBase that uses GPU device 1.

    This class is designed to run on the second GPU (index 1) if available.
    It inherits all functionality from TextRedactorBase.
    """
    def __init__(self):
        """
        Initialize TextRedactor_1 with GPU device 1.
        """
        super().__init__(device=0)


class TextRedactor_1(TextRedactorBase):
    """
    A specific implementation of TextRedactorBase that uses GPU device 1.

    This class is designed to run on the second GPU (index 1) if available.
    It inherits all functionality from TextRedactorBase.
    """
    def __init__(self):
        """
        Initialize TextRedactor_1 with GPU device 1.
        """
        super().__init__(device=1)


def redact_files_with_model(model_instance: TextRedactorBase, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for transcription in files:
        start_time = time.time()
        anonymized_transcription = model_instance.predict(None, [transcription])
        processing_time = time.time() - start_time
        results.append({
            'original': transcription,
            'redacted': anonymized_transcription[0],
            'processing_time': processing_time
        })
    return results


def load_models_on_gpu() -> tuple[PythonModel, PythonModel]:
    os.makedirs('redaction_pipeline1', exist_ok=True)
    os.makedirs('redaction_pipeline2', exist_ok=True)

    with mlflow.start_run() as run0:
        mlflow.pyfunc.log_model(
            artifact_path="redaction_pipeline1",
            python_model=TextRedactor_0(),
            artifacts={"redaction_pipeline1": "./redaction_pipeline1"}
        )

    model_uri0 = f"runs:/{run0.info.run_id}/redaction_pipeline1"
    model_0 = mlflow.pyfunc.load_model(model_uri0)

    with mlflow.start_run() as run1:
        mlflow.pyfunc.log_model(
            artifact_path="redaction_pipeline2",
            python_model=TextRedactor_1(),
            artifacts={"redaction_pipeline2": "./redaction_pipeline2"}
        )

    model_uri1 = f"runs:/{run1.info.run_id}/redaction_pipeline2"
    model_1 = mlflow.pyfunc.load_model(model_uri1)

    return model_0, model_1


def load_credentials(filepath: str) -> Dict[str, str]:
    with open(filepath, "r") as f:
        creds = json.load(f)
    return creds


def main():
    start_time = time.time()

    model_0, model_1 = load_models_on_gpu()

    creds = load_credentials("/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/CallVoltMasterRepo/RepeatCallers/credentials/redact.json")

    transcriptions = pd.read_parquet(creds['input_path'])
    file_names = transcriptions['transcription'].tolist()
    half_size = len(file_names) // 2
    files_gpu_0 = file_names[:half_size]
    files_gpu_1 = file_names[half_size:]

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_gpu0 = executor.submit(redact_files_with_model, model_0, files_gpu_0)
        future_gpu1 = executor.submit(redact_files_with_model, model_1, files_gpu_1)

        results_gpu0 = future_gpu0.result()
        results_gpu1 = future_gpu1.result()

    df_gpu0 = pd.DataFrame(results_gpu0)
    df_gpu1 = pd.DataFrame(results_gpu1)
    final_df = pd.concat([df_gpu0, df_gpu1], ignore_index=True)

    total_time = time.time() - start_time

    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Total files processed: {len(final_df)}")
    print(f"Average time per file: {total_time / len(final_df):.2f} seconds")
    print(f"Max processing time for a single file: {final_df['processing_time'].max():.2f} seconds")
    print(f"Min processing time for a single file: {final_df['processing_time'].min():.2f} seconds")

    try:
        final_df.to_parquet(creds['output_path'], index=False)
        print(f"DataFrame successfully saved to: {creds['output_path']}")
    except Exception as e:
        print(f"Error: Failed to save DataFrame to {creds['output_path']}. {str(e)}")


if __name__ == "__main__":
    main()
