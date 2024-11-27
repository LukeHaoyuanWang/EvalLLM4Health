import os
import whisper

def transcribe_folder(input_folder, model_name="base", language="en", output_folder="transcripts"):
    """
    Transcribes each audio file in the input folder using Whisper and saves the transcript as a text file.
    
    Parameters:
        input_folder (str): Path to the folder containing audio files.
        model_name (str): Whisper model to use ("tiny", "base", "small", "medium", "large").
        language (str): Language code for the transcription (e.g., "en" for English).
        output_folder (str): Path to the folder where transcriptions will be saved.
    """
    # Load the Whisper model
    model = whisper.load_model(model_name)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate over each file in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        # Check if the file is an audio file 
        if filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg')):
            # Transcribe the audio file
            result = model.transcribe(file_path, language=language)
            
            # Generate output filename
            output_filename = os.path.splitext(filename)[0] + "_raw_transcript.txt"
            output_path = os.path.join(output_folder, output_filename)
            
            # Save the transcription to a text file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result['text'])
                
            print(f"Transcribed '{filename}' and saved as '{output_filename}'")

    print("Transcription completed for all audio files.")