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
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta

class Transcribe:
    def __init__(self, device_index: int):
        """
        Initializes the Transcribe class with the specified GPU device index.
        
        Args:
            device_index (int): The index of the GPU to use for transcription.
        """
        self.device_index = device_index
        self.transcription_model = None
        self.load_context()

    def load_context(self) -> None:
        """
        Loads the Whisper model onto the specified GPU device with the float16 precision.
        """
        self.transcription_model = WhisperModel("medium.en", 
                                                device="cuda", 
                                                device_index=self.device_index, 
                                                compute_type="float16")

    def predict(self, model_input: str) -> List[Dict[str, Any]]:
        """
        Transcribes the given audio file and returns the transcription segments.
        
        Args:
            model_input (str): Path to the audio file to transcribe.
        
        Returns:
            List[Dict[str, Any]]: A list of transcription segments, where each segment is 
                                   represented as a dictionary with 'start', 'end', 'text', 
                                   and 'no_speech_probability' keys.
        """
        try:
            segments, _ = self.transcription_model.transcribe(
                model_input,
                beam_size=5,
                language="en",
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            return segments
        except Exception as e:
            print(f"Error in transcription on GPU {self.device_index}: {str(e)}")
            return []

def transcribe_files(audio_files: List[str], device_index: int) -> List[Dict[str, Any]]:
    """
    Transcribes a list of audio files using a specific GPU device and returns their transcriptions.
    
    Args:
        audio_files (List[str]): List of paths to audio files to transcribe.
        device_index (int): Index of the GPU to use for transcription.
    
    Returns:
        List[Dict[str, Any]]: A list of transcription results, where each result includes
                               the file path, transcription details, and the time taken for transcription.
    """
    model = Transcribe(device_index)
    transcriptions = []
    for audio_file in audio_files:
        print(f"Transcribing file {audio_file} on GPU {device_index}...")
        start_time = time.time()
        segments = model.predict(audio_file)
        end_time = time.time()
        transcription_time = end_time - start_time

        file_transcription = [{
            'start': segment.start,
            'end': segment.end,
            'text': segment.text,
            'no_speech_probability': segment.no_speech_prob
        } for segment in segments]

        transcriptions.append({
            'file': audio_file,
            'transcription': file_transcription,
            'transcription_time': transcription_time
        })
    return transcriptions

def main() -> None:
    """
    Main function to handle transcription of audio files using GPUs and save the results.
    """
    with open("/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/CallVoltMasterRepo/RepeatCallers/credentials/transcription.json", "r") as f:
        creds = json.load(f)

    today = datetime.now().date() - timedelta(days=2)
    today_str = today.strftime('%Y_%m_%d')
    dbfs_path = creds['input_path']
    file_names = [os.path.join(dbfs_path, f) for f in os.listdir(dbfs_path)]
    half_size = len(file_names) // 2

    files_gpu_0 = file_names[:half_size]
    files_gpu_1 = file_names[half_size:]

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(transcribe_files, files_gpu_0, 0),
            executor.submit(transcribe_files, files_gpu_1, 1)
        ]
        results = [future.result() for future in futures]
        executor.shutdown()

    # Combine results from both GPUs
    all_transcriptions = results[0] + results[1]
    transcription_df = pd.DataFrame(all_transcriptions)
    # Construct the file path for saving
    save_path = creds['output_path']

    # Save the DataFrame to Parquet format
    try:
        transcription_df.to_parquet(save_path, index=False)
        print(f"DataFrame successfully saved to: {save_path}")
    except Exception as e:
        print(f"Error: Failed to save DataFrame to {save_path}. Exception: {e}")

if __name__ == "__main__":
    main()
