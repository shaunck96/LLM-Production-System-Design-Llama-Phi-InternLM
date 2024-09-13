import torch
import torch.distributed as dist
import os
import logging
import pandas as pd
from faster_whisper import *
import time
from torch import device
import os
import pandas as pd
import time
from faster_whisper import WhisperModel
from datetime import datetime, timedelta
import mlflow
from concurrent.futures import ThreadPoolExecutor
from typing import Union, List, Dict, Tuple
import json

class Transcribe_0(mlflow.pyfunc.PythonModel):
    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Load the transcription model and set up logging.

        Args:
            context (mlflow.pyfunc.PythonModelContext): The context for the model.
        """
        self.transcription_model = WhisperModel("medium.en",
                                                device="cuda",
                                                device_index=0,
                                                compute_type="float16")
        self.logger = logging.getLogger(f"{__name__}.Transcribe_0")

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: Union[str, List[Dict]]) -> List[Dict[str, Union[float, str]]]:
        """
        Transcribe the audio file and return the transcriptions.

        Args:
            context (mlflow.pyfunc.PythonModelContext): The context for the model.
            model_input (Union[str, List[Dict]]): File path of the audio file.

        Returns:
            List[Dict[str, Union[float, str]]]: A list of dictionaries with transcription details.
        """
        try:
            if isinstance(model_input, str):
                # Assume model_input is a file path
                if not os.path.exists(model_input):
                    raise FileNotFoundError(f"File not found: {model_input}")
                
                segments, _ = self.transcription_model.transcribe(
                    model_input,
                    beam_size=5,
                    language="en",
                    condition_on_previous_text=False,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
            else:
                raise ValueError("Expected a file path string as input")
            
            segments = list(segments)
            segment_mapping_dict = {
                2: 'start',
                3: 'end',
                4: 'text',
                9: 'no_speech_probability'
            }
            
            transcriptions = []
            for index in range(len(segments)):
                transcription_dict = {}
                for segment_index in list(segment_mapping_dict.keys()):
                    transcription_dict[segment_mapping_dict[segment_index]] = segments[index][segment_index]
                transcriptions.append(transcription_dict)
            
            return transcriptions
        
        except Exception as e:
            self.logger.error(f'Error in transcription: {str(e)}')
            return []

class Transcribe_1(mlflow.pyfunc.PythonModel):
    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Load the transcription model and set up logging.

        Args:
            context (mlflow.pyfunc.PythonModelContext): The context for the model.
        """
        self.transcription_model = WhisperModel("medium.en",
                                                device="cuda",
                                                device_index=1,
                                                compute_type="float16")
        self.logger = logging.getLogger(f"{__name__}.Transcribe_1")

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: Union[str, List[Dict]]) -> List[Dict[str, Union[float, str]]]:
        """
        Transcribe the audio file and return the transcriptions.

        Args:
            context (mlflow.pyfunc.PythonModelContext): The context for the model.
            model_input (Union[str, List[Dict]]): File path of the audio file.

        Returns:
            List[Dict[str, Union[float, str]]]: A list of dictionaries with transcription details.
        """
        try:
            if isinstance(model_input, str):
                # Assume model_input is a file path
                if not os.path.exists(model_input):
                    raise FileNotFoundError(f"File not found: {model_input}")
                
                segments, _ = self.transcription_model.transcribe(
                    model_input,
                    beam_size=5,
                    language="en",
                    condition_on_previous_text=False,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
            else:
                raise ValueError("Expected a file path string as input")
            
            segments = list(segments)
            segment_mapping_dict = {
                2: 'start',
                3: 'end',
                4: 'text',
                9: 'no_speech_probability'
            }
            
            transcriptions = []
            for index in range(len(segments)):
                transcription_dict = {}
                for segment_index in list(segment_mapping_dict.keys()):
                    transcription_dict[segment_mapping_dict[segment_index]] = segments[index][segment_index]
                transcriptions.append(transcription_dict)
            
            return transcriptions
        
        except Exception as e:
            self.logger.error(f'Error in transcription: {str(e)}')
            return []

def transcribe_files_with_model(model_instance: mlflow.pyfunc.PythonModel, files: List[str]) -> List[Dict[str, Union[str, List[Dict[str, Union[float, str]]], float]]]:
    """
    Transcribe a list of audio files using a specified model instance.

    Args:
        model_instance (mlflow.pyfunc.PythonModel): An instance of the transcription model.
        files (List[str]): A list of file paths to audio files.

    Returns:
        List[Dict[str, Union[str, List[Dict[str, Union[float, str]]], float]]]: A list of dictionaries, each containing the file path, the transcription result, and the processing time.
    """
    results = []
    for audio_file in files:
        start_time = time.time()
        transcription = model_instance.predict(audio_file)
        results.append({"audio_file":audio_file,
                        "transcription":transcription,
                        "processing_time":time.time() - start_time})
    return results

def load_models_on_gpu() -> Tuple[mlflow.pyfunc.PythonModel, mlflow.pyfunc.PythonModel]:
    """
    Load and return two transcription models from MLflow, one for each GPU.

    Returns:
        Tuple[mlflow.pyfunc.PythonModel, mlflow.pyfunc.PythonModel]: A tuple containing two instances of the transcription models, one for each GPU.
    """
    os.makedirs('transcription_pipeline1', exist_ok=True)
    os.makedirs('transcription_pipeline2', exist_ok=True)

    with mlflow.start_run() as run0:
        mlflow.pyfunc.log_model(artifact_path="transcription_pipeline1",
                                python_model=Transcribe_0(),
                                artifacts={"transcription_pipeline1": "./transcription_pipeline1"})
    
    model_uri0 = "runs:/{run}/transcription_pipeline1".format(run = run0.info.run_id)

    with mlflow.start_run() as run1:
        mlflow.pyfunc.log_model(artifact_path="transcription_pipeline2",
                                python_model=Transcribe_1(),
                                artifacts={"transcription_pipeline2": "./transcription_pipeline2"})
    
    model_uri1 = "runs:/{run}/transcription_pipeline2".format(run = run1.info.run_id)
    model_0 = mlflow.pyfunc.load_model(model_uri0)
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

# Load models and credentials
model_0, model_1 = load_models_on_gpu()

creds = load_credentials("/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/CallVoltMasterRepo/RepeatCallers/credentials/transcription.json")

input_path = creds['input_path']
save_path = creds['output_path']
file_names = [os.path.join(input_path, f) for f in os.listdir(input_path)]
print("Number of Files Transcribed: " + str(len(file_names)))

half_size = len(file_names) // 2
files_gpu_0 = file_names[:half_size]
files_gpu_1 = file_names[half_size:]
results_gpu0 = []
results_gpu1 = []

start_time = time.time()

with ThreadPoolExecutor(max_workers=2) as executor:
    future_gpu0 = executor.submit(transcribe_files_with_model, model_0, files_gpu_0)
    future_gpu1 = executor.submit(transcribe_files_with_model, model_1, files_gpu_1)

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
    final_df.to_parquet(save_path, index=False)
    print(f"DataFrame successfully saved to: {save_path}")
except Exception as e:
    print(f"Error: Failed to save DataFrame to {save_path}. {str(e)}")
