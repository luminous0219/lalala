Build the Docker image

open command prompt

cd lalala\asr

docker build -t asr-api .


docker run -p 8001:8001 -e TEMP_DIR=your_temp_directory asr-api


open another command prompt

curl http://localhost:8001/ping

output: {"message":"pong"}

curl -F "file=@/path/to/your/sample.mp3" http://localhost:8001/asr

for example,
C:\Users\ml178>curl -F "file=@C:\Users\ml178\OneDrive\Desktop\asr\cv-valid-dev16k\sample-000000.mp3" http://localhost:8001/asr

output: {"transcription":"BE CAREFUL THAT YOU PROGNOSTICATIONS SAID THE STRANGER","duration":"5.1"}

Running cv-decode.py

cd lalala\asr

pip install -r requirements.txt

python cv-decode.py --csv_path cv-valid-dev.csv --audio_dir cv-valid-dev16k/ --api_url http://localhost:8001/asr --log_file transcription_errors.log

output:
Estimated time remaining: 4441.6 seconds.
Processed 18/4076 files.
Estimated time remaining: 4470.9 seconds.
Processed 19/4076 files.
Estimated time remaining: 4381.1 seconds.
Processed 20/4076 files.
.
.
.

Transcription completed and saved to CSV.