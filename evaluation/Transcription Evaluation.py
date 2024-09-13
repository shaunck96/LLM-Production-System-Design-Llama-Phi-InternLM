# Databricks notebook source
!pip install faster-whisper --upgrade
!pip show faster-whisper
!pip install jiwer
!pip install python-Levenshtein
!pip install scikit-learn
!pip install nltk
!pip install jinja2


# COMMAND ----------

import torch
import os
import pandas as pd
import time
from faster_whisper import WhisperModel
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
import gc
from jiwer import wer, cer
from Levenshtein import distance as levenshtein_distance
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader
import os
from IPython.display import display, HTML
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import precision_score, recall_score

class Transcribe:
    def __init__(self, model_size, device_index):
        self.device_index = device_index
        self.transcription_model = None
        self.model_size = model_size
        self.load_context()

    def load_context(self):
        # Load the Whisper model
        self.transcription_model = WhisperModel(self.model_size,
                                                device="cuda",
                                                device_index=self.device_index,
                                                compute_type="float16")

    def predict(self, model_input):
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

    def unload_model(self):
        # Unload model and clear memory
        del self.transcription_model
        torch.cuda.empty_cache()  # Clear unused memory on GPU
        gc.collect()  # Trigger garbage collection for CPU memory

def transcribe_files(audio_files, model_size, device_index):
    model = Transcribe(model_size, device_index)
    transcriptions = []
    for audio_file in audio_files:
        print(f"Transcribing file {audio_file} on GPU {device_index} using {model_size} model...")
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

    # After all files are processed, unload the model to free memory
    model.unload_model()

    return transcriptions

def merger(anonymized_list):
    return " ".join([trans['text'] for trans in anonymized_list])


def calculate_metrics(merged_df):
    # Pre-store the columns in NumPy arrays for faster access
    transcriptions_medium = merged_df['transcription_medium'].values
    transcriptions_large = merged_df['transcription_large'].values

    # Calculate WER and CER for all rows using list comprehensions (vectorized)
    merged_df['WER'] = [wer(t1, t2) for t1, t2 in zip(transcriptions_medium, transcriptions_large)]
    merged_df['CER'] = [cer(t1, t2) for t1, t2 in zip(transcriptions_medium, transcriptions_large)]

    # Summarize the WER and CER
    final_wer = merged_df['WER'].mean()
    final_cer = merged_df['CER'].mean()

    print(f"Final summarized WER: {final_wer}")
    print(f"Final summarized CER: {final_cer}")

    return merged_df, final_wer, final_cer


def compute_precision_recall(transcription_1, transcription_2):
    # Tokenize words and compute precision and recall
    # Simplified for illustration
    tokens_1 = set(transcription_1.split())
    tokens_2 = set(transcription_2.split())
    common = len(tokens_1 & tokens_2)
    precision = common / len(tokens_1) if tokens_1 else 0
    recall = common / len(tokens_2) if tokens_2 else 0
    return precision, recall

def generate_report(merged_df, final_wer, final_cer, levenshtein_mean, levenshtein_median, levenshtein_std,
                    precision_mean, precision_median, precision_std, recall_mean, recall_median, recall_std,
                    f1_mean, f1_median, f1_std, bleu_summary):
    
    # Set up the Jinja2 environment
    env = Environment(loader=FileSystemLoader(searchpath='./'))
    template = env.get_template('report_template.html')

    # Get a sample of 5 rows
    sample_rows = merged_df[['transcription_medium', 'transcription_large', 'BLEU']].head(5).to_dict('records')

    # Render the HTML report
    report_html = template.render(
        final_wer=final_wer,
        final_cer=final_cer,
        levenshtein_mean=levenshtein_mean,
        levenshtein_median=levenshtein_median,
        levenshtein_std=levenshtein_std,
        precision_mean=precision_mean,
        precision_median=precision_median,
        precision_std=precision_std,
        recall_mean=recall_mean,
        recall_median=recall_median,
        recall_std=recall_std,
        f1_mean=f1_mean,
        f1_median=f1_median,
        f1_std=f1_std,
        bleu_summary=bleu_summary.to_string(),
        sample_rows=sample_rows
    )

    # Save the report to an HTML file
    with open('transcription_metrics_report.html', 'w') as f:
        f.write(report_html)

    print("Report generated and saved as 'transcription_metrics_report.html'.")

today = datetime.now().date() - timedelta(days=14)
today_str = today.strftime('%Y_%m_%d')
base_dir = "/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/audio_files"
dbfs_path = f"/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/audio_files/{today_str}"

file_names = [os.path.join(dbfs_path, f) for f in os.listdir(dbfs_path)]
half_size = len(file_names) // 2

files_gpu_0 = file_names[:half_size]
files_gpu_1 = file_names[half_size:]

with ProcessPoolExecutor(max_workers=2) as executor:
    futures_medium = [
        executor.submit(transcribe_files, files_gpu_0, "medium.en", 0),
        executor.submit(transcribe_files, files_gpu_1, "medium.en", 1)
    ]
    futures_large = [
        executor.submit(transcribe_files, files_gpu_0, "large", 0),
        executor.submit(transcribe_files, files_gpu_1, "large", 1)
    ]
    results_medium = [future.result() for future in futures_medium]
    results_large = [future.result() for future in futures_large]
    executor.shutdown()

# Combine results from both GPUs for each model
all_transcriptions_medium = results_medium[0] + results_medium[1]
all_transcriptions_large = results_large[0] + results_large[1]

# Convert the results into DataFrames
transcription_df_medium = pd.DataFrame(all_transcriptions_medium)
transcription_df_large = pd.DataFrame(all_transcriptions_large)


# Merge the DataFrames based on 'file' column
merged_df = transcription_df_medium.merge(transcription_df_large, on='file', suffixes=('_medium', '_large'))
merged_df['transcription_medium'] = merged_df['transcription_medium'].apply(merger)
merged_df['transcription_large'] = merged_df['transcription_large'].apply(merger)

merged_df.to_csv(r'/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/Production_Code/Deployment Code/evaluation/transcription_eval.csv')

# Use the optimized function
merged_df, final_wer, final_cer = calculate_metrics(merged_df)

merged_df['Levenshtein'] = merged_df.apply(lambda row: levenshtein_distance(row['transcription_medium'], row['transcription_large']), axis=1)

# Calculate summary statistics for Levenshtein distance
levenshtein_mean = merged_df['Levenshtein'].mean()
levenshtein_median = merged_df['Levenshtein'].median()
levenshtein_std = merged_df['Levenshtein'].std()

print(f"Levenshtein Mean: {levenshtein_mean}")
print(f"Levenshtein Median: {levenshtein_median}")
print(f"Levenshtein Standard Deviation: {levenshtein_std}")

# Plot the distribution of Levenshtein distances
plt.hist(merged_df['Levenshtein'], bins=30, color='blue', edgecolor='black')
plt.title('Distribution of Levenshtein Distances')
plt.xlabel('Levenshtein Distance')
plt.ylabel('Frequency')
plt.show()

merged_df['Precision'], merged_df['Recall'] = zip(*merged_df.apply(lambda row: compute_precision_recall(row['transcription_medium'], row['transcription_large']), axis=1))

# Calculate summary statistics for Precision and Recall
precision_mean = merged_df['Precision'].mean()
precision_median = merged_df['Precision'].median()
precision_std = merged_df['Precision'].std()

recall_mean = merged_df['Recall'].mean()
recall_median = merged_df['Recall'].median()
recall_std = merged_df['Recall'].std()

print(f"Precision Mean: {precision_mean}")
print(f"Precision Median: {precision_median}")
print(f"Precision Standard Deviation: {precision_std}")

print(f"Recall Mean: {recall_mean}")
print(f"Recall Median: {recall_median}")
print(f"Recall Standard Deviation: {recall_std}")

merged_df['F1_Score'] = 2 * (merged_df['Precision'] * merged_df['Recall']) / (merged_df['Precision'] + merged_df['Recall'])

# Calculate summary statistics for F1 Score
f1_mean = merged_df['F1_Score'].mean()
f1_median = merged_df['F1_Score'].median()
f1_std = merged_df['F1_Score'].std()

print(f"F1 Score Mean: {f1_mean}")
print(f"F1 Score Median: {f1_median}")
print(f"F1 Score Standard Deviation: {f1_std}")

merged_df['BLEU'] = merged_df.apply(lambda row: sentence_bleu([row['transcription_medium'].split()], row['transcription_large'].split()), axis=1)

# Summary statistics for BLEU scores
bleu_summary = merged_df['BLEU'].describe()

# Print the summary statistics
print("BLEU Score Summary Statistics:")
print(bleu_summary)

# Print a sample of rows with BLEU scores
print("\nSample of Transcription Comparisons and BLEU Scores:")
print(merged_df[['transcription_medium', 'transcription_large', 'BLEU']].head(5))

# Use the generate_report function to create the report
generate_report(merged_df, final_wer, final_cer, levenshtein_mean, levenshtein_median, levenshtein_std,
                precision_mean, precision_median, precision_std, recall_mean, recall_median, recall_std,
                f1_mean, f1_median, f1_std, bleu_summary)

# Load and display the HTML file
with open('transcription_metrics_report.html', 'r') as file:
    report_html = file.read()

display(HTML(report_html))

output_csv_path = '/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/CallVoltMasterRepo/RepeatCallers/evaluation/transcriptions/transcription_metrics_evaluation.csv'
merged_df.to_csv(output_csv_path)
