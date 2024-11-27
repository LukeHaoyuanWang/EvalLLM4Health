import re
import os
from nltk.tokenize import sent_tokenize

########################################## Truncate Text ###############################################
def truncate_transcript(text, max_words=1000):
    sentences = sent_tokenize(text)
    current_count = 0
    batch = []
    batches = []
    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_length = len(sentence_words)
        # If adding this sentence exceeds the limit, store the current batch and start a new one
        if current_count + sentence_length > max_words:
            if batch:
                batches.append(' '.join(batch))
                batch = []
                current_count = 0
        batch.extend(sentence_words)
        current_count += sentence_length
    # Add the last batch if there's any remaining
    if batch:
        batches.append(' '.join(batch))
    return batches

########################################## Main Function ###############################################
def annotate_speaker_roles(transcript_batch, client,openai_chat_model):
    additional_instructions = """
    Below is a transcript of a conversation between a doctor and a patient. Try to separate the sentences and annotate the roles of the speakers, either doctor or patient:
    """
    user_content = additional_instructions + '\n\n' + transcript_batch
    chat_completion = client.chat.completions.create(
        model=openai_chat_model,  # specify the correct model here
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant tasked with annotating the roles of speakers in the following transcript."
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
    )
    # Extract the response content
    return chat_completion.choices[0].message.content

def process_transcripts(input_folder, output_folder, client, openai_chat_model="gpt-4"):
    """
    Process all transcript files in the input folder to annotate speaker roles and save the results
    in the output folder.

    Args:
        input_folder (str): Path to the folder containing transcript files.
        output_folder (str): Path to the folder where annotated files will be saved.
        client: OpenAI client object to interact with OpenAI's chat models.
        openai_chat_model (str): The model name to be used for OpenAI chat completion.

    Returns:
        None: Annotated files are saved to the output folder.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):  # Assuming transcripts are stored as .txt files
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Read the transcript file
            with open(input_path, 'r') as file:
                transcript = file.read()

            # Process the transcript using the helper functions
            transcript_batches = truncate_transcript(transcript, max_words=1000)
            annotated_batches = [
                annotate_speaker_roles(batch, client, openai_chat_model)
                for batch in transcript_batches
            ]

            # Combine the annotated batches into a single annotated transcript
            annotated_transcript = "\n".join(annotated_batches)

            # Save the annotated transcript to the output folder
            with open(output_path, 'w') as file:
                file.write(annotated_transcript)

            print(f"Processed and saved: {filename}")
