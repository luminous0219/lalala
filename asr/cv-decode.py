import requests
import pandas as pd
import os
import time
import logging
import argparse

def main(csv_path, audio_dir, api_url, log_file):
    # Load CSV file
    df = pd.read_csv(csv_path)
    
    # Add the "generated_text" column if it doesn't exist
    if 'generated_text' not in df.columns:
        df['generated_text'] = ''
    
    # Set up logging for errors
    logging.basicConfig(filename=log_file, level=logging.ERROR)
    
    # Total number of files to process
    total_files = len(df)
    processed_files = 0
    start_time = time.time()
    
    # Loop through each row in the CSV file
    try:
        for index, row in df.iterrows():
            # Construct the full path to the audio file
            audio_file = os.path.join(audio_dir, os.path.basename(row['filename']))
            
            # Skip if the file doesn't exist
            if not os.path.exists(audio_file):
                print(f"File {audio_file} not found, skipping.")
                continue

            try:
                # Start timing the transcription process for this file
                file_start_time = time.time()

                # Send request to ASR API with a timeout of 60 seconds
                with open(audio_file, 'rb') as f:
                    files = {'file': (row['filename'], f, 'audio/mp3')}
                    response = requests.post(api_url, files=files, timeout=60)
                
                # Handle the API response
                if response.status_code == 200:
                    result = response.json()
                    # Add the transcription to the 'generated_text' column
                    df.at[index, 'generated_text'] = result['transcription']
                    processed_files += 1

                    # Save the updated CSV file after each transcription
                    df.to_csv(csv_path, index=False)

                    # Calculate progress and estimated time remaining
                    elapsed_time = time.time() - start_time
                    avg_time_per_file = elapsed_time / processed_files
                    remaining_files = total_files - processed_files
                    estimated_remaining_time = remaining_files * avg_time_per_file

                    # Print progress and time estimates
                    print(f"Processed {processed_files}/{total_files} files.")
                    print(f"Estimated time remaining: {estimated_remaining_time:.1f} seconds.")
                
                else:
                    logging.error(f"Failed to transcribe {audio_file}. Status code: {response.status_code}")
                    print(f"Failed to transcribe file {audio_file}")
            
            except requests.exceptions.RequestException as e:
                logging.error(f"Error processing file {audio_file}: {e}")
                print(f"Error processing file {audio_file}, skipping.")

    except KeyboardInterrupt:
        print("Process interrupted by user. Saving partial results...")
    
    print("Transcription completed and saved to CSV.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio files using ASR API.")
    parser.add_argument('--csv_path', type=str, default="cv-valid-dev.csv", help='Path to the CSV file.')
    parser.add_argument('--audio_dir', type=str, default="cv-valid-dev16k/", help='Directory containing audio files.')
    parser.add_argument('--api_url', type=str, default="http://localhost:8001/asr", help='ASR API URL.')
    parser.add_argument('--log_file', type=str, default="transcription_errors.log", help='Log file for errors.')
    
    args = parser.parse_args()
    
    main(args.csv_path, args.audio_dir, args.api_url, args.log_file)
