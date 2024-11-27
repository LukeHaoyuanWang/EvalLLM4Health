import re
import os
from openai import OpenAI
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize

load_dotenv()
client = OpenAI(
    api_key = 'Your Key',
)

#################################### Truncate Text ########################################
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


#################################### Deidentify Names ########################################
def identify_names(transcript_batch, client, model="gpt-4"):
    """
    Extracts all the possible names from the text. Uses the specified model to process the text.

    Parameters:
    - transcript_batch (str): The text batch to analyze.
    - client: The OpenAI API client instance.
    - model (str): The model to use for processing (default: "gpt-4").

    Returns:
    - set: A set of extracted names. If no names are found, returns an empty set.
    """
    instructions = """
    Extract all the possible names from the text. Return all names mentioned in the text as it is separated by commas. If there are no names, return the word None. Note some names are in lower case. 
    """
    user_content = instructions + '\n\n' + transcript_batch
    chat_completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant tasked with identifying names in the transcript."
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
    )
    # Extract the response content, assuming it returns names as a comma-separated list
    response = chat_completion.choices[0].message.content
    names = {name.strip() for name in response.split(',') if name.strip() != "None"}
    return names

def replace_names_with_identifiers(batches, client, model="gpt-4"):
    """
    Replace names in text batches with unique identifiers.

    Parameters:
    - batches (list of str): A list of text batches.
    - client: The OpenAI API client instance.
    - model (str): The model to use for processing (default: "gpt-4").

    Returns:
    - updated_batches (list of str): The text batches with names replaced by unique identifiers.
    - name_mapping (dict): A dictionary mapping original names to their identifiers.
    """
    unique_names = set()
    
    # Step 1: Collect all unique names from all batches using OpenAI API
    for batch in batches:
        unique_names.update(identify_names(batch, client, model=model))

    # Step 2: Assign unique identifiers to each name
    name_mapping = {name: f"NAME{i+1}" for i, name in enumerate(sorted(unique_names))}

    # Step 3: Replace names in each batch with their respective identifier while preserving formatting
    updated_batches = []
    for batch in batches:
        # Preserve the line breaks and separators within each batch by splitting and re-joining
        lines = batch.splitlines()  # Split into lines to preserve structure
        updated_lines = []
        
        for line in lines:
            # Replace names in each line while preserving the line structure
            updated_line = line
            for name, identifier in name_mapping.items():
                updated_line = re.sub(rf'\b{name}\b', identifier, updated_line)
            updated_lines.append(updated_line)
        
        # Join the updated lines back into the batch with line breaks
        updated_batch = "\n".join(updated_lines)
        updated_batches.append(updated_batch)
    
    return updated_batches, name_mapping

#################################### Deidentify Locations ########################################
def identify_locations(transcript_batch, client, model="gpt-4"):
    """
    Identify all addresses and locations in a transcript batch.

    Args:
        transcript_batch (str): The transcript text to process.
        client (object): The client object for interacting with the API.
        model (str, optional): The model to use for processing. Defaults to "gpt-4".

    Returns:
        set: A set of unique locations identified in the transcript. If no locations are found, returns an empty set.
    """
    instructions = """
    Below is a transcript of a conversation between a doctor and a patient. Identify all the addresses and locations in the conversation. Return the addresses and locations as a comma-separated list. If there are no addresses, only return the word None.
    """
    user_content = instructions + '\n\n' + transcript_batch
    chat_completion = client.chat.completions.create(
        model=model,  # Allow dynamic specification of the model
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant tasked with identifying locations in the transcript."
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
    )
    # Extract the response content, assuming it returns locations as a comma-separated list
    response = chat_completion.choices[0].message.content
    locations = {location.strip() for location in response.split(',') if location.strip() != "None"}
    return locations

def replace_locations_with_identifiers(batches, client, model="gpt-4"):
    """
    Replace all locations in the given text batches with unique identifiers.

    Args:
        batches (list): A list of transcript text batches.
        client (object): The client object for interacting with the API.
        model (str, optional): The model to use for identifying locations. Defaults to "gpt-4".

    Returns:
        tuple: A tuple containing:
            - updated_batches (list): List of updated transcript batches with locations replaced by identifiers.
            - location_mapping (dict): Mapping of original locations to their unique identifiers.
    """
    unique_locations = set()
    
    # Step 1: Collect all unique locations from all batches using the identify_locations function
    for batch in batches:
        unique_locations.update(identify_locations(batch, client, model=model))

    # Step 2: Assign unique identifiers to each location
    location_mapping = {location: f"LOCATION{i+1}" for i, location in enumerate(unique_locations)}

    # Step 3: Replace locations in each batch with their respective identifier while preserving formatting
    updated_batches = []
    for batch in batches:
        # Preserve the line breaks and separators within each batch by splitting and re-joining
        lines = batch.splitlines()  # Split into lines to preserve structure
        updated_lines = []
        
        for line in lines:
            # Replace locations in each line while preserving the line structure
            updated_line = line
            for location, identifier in location_mapping.items():
                updated_line = re.sub(rf'\b{re.escape(location)}\b', identifier, updated_line)
            updated_lines.append(updated_line)
        
        # Join the updated lines back into the batch with line breaks
        updated_batch = "\n".join(updated_lines)
        updated_batches.append(updated_batch)
    
    return updated_batches, location_mapping

#################################### Deidentify Dates ########################################
def identify_dates(transcript_batch, client, model="gpt-4"):
    """
    Identifies all dates mentioned in a transcript of a conversation between a doctor and a patient.

    Args:
        transcript_batch (str): The transcript text to analyze for dates.
        client (object): The client object for interacting with the language model API.
        model (str, optional): The language model to use for processing. Defaults to "gpt-4".

    Returns:
        set: A set of identified dates from the transcript. If no dates are found, returns an empty set.
    """
    # Define the instructions for the model to extract dates from the transcript
    instructions = """
    Below is a transcript of a conversation between a doctor and a patient. Identify all the dates in the conversation. 
    Return the dates as a comma-separated list. If there are no dates, return only the word 'None'.
    """

    # Combine the instructions and transcript into a single input for the model
    user_content = instructions + '\n\n' + transcript_batch

    # Send the input to the language model API for processing
    chat_completion = client.chat.completions.create(
        model=model,  # Use the specified model (default is "gpt-4")
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant tasked with identifying dates in the transcript."
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
    )

    # Extract the model's response, expected as a comma-separated list of dates or "None"
    response = chat_completion.choices[0].message.content

    # Parse the response to extract dates, ignoring the word "None"
    dates = {date.strip() for date in response.split(',') if date.strip() != "None"}

    return dates  # Return the set of unique dates

def replace_dates_with_identifiers(batches, client, model="gpt-4"):
    """
    Replaces all dates in text batches with unique identifiers while preserving formatting.

    Args:
        batches (list of str): List of text batches (e.g., transcripts) to process.
        client (object): Client object for interacting with the language model API.
        model (str, optional): The language model to use for processing. Defaults to "gpt-4".

    Returns:
        tuple:
            - updated_batches (list of str): List of text batches with dates replaced by unique identifiers.
            - date_mapping (dict): A mapping of original dates to their unique identifiers.
    """
    unique_dates = set()

    # Step 1: Collect all unique dates from all batches using the language model
    for batch in batches:
        # Use the specified model for identifying dates
        unique_dates.update(identify_dates(batch, client, model=model))

    # Step 2: Assign unique identifiers to each date
    date_mapping = {date: f"DATE{i+1}" for i, date in enumerate(unique_dates)}

    # Step 3: Replace dates in each batch with their respective identifier while preserving formatting
    updated_batches = []
    for batch in batches:
        # Split batch into lines to preserve its original structure
        lines = batch.splitlines()
        updated_lines = []

        for line in lines:
            # Replace dates in the current line with their unique identifiers
            updated_line = line
            for date, identifier in date_mapping.items():
                # Use regex to replace only exact matches of the date
                updated_line = re.sub(rf'\b{re.escape(date)}\b', identifier, updated_line)
            updated_lines.append(updated_line)

        # Join updated lines back into a single string
        updated_batch = "\n".join(updated_lines)
        updated_batches.append(updated_batch)

    return updated_batches, date_mapping

#################################### One Function ########################################
def deidentify_transcripts(
    input_folder_path,
    output_folder_path,
    deidentify_name=True,
    deidentify_location=False,
    deidentify_date=False,
    model="gpt-4",
    client=None,
    return_mapping=False
):
    """
    De-identify all transcript files in a folder by replacing sensitive information such as names, locations, and dates.

    Parameters:
    - input_folder_path (str): Path to the input folder containing transcript files.
    - output_folder_path (str): Path to the folder to save the de-identified transcripts.
    - deidentify_name (bool): Whether to de-identify names (default: True).
    - deidentify_location (bool): Whether to de-identify locations (default: False).
    - deidentify_date (bool): Whether to de-identify dates (default: False).
    - model (str): The model to use for processing (default: "gpt-4").
    - client: The OpenAI API client instance.
    - return_mapping (bool): Whether to return mappings for names, locations, and dates (default: False).

    Returns:
    - dict (optional): If return_mapping is True, returns a dictionary with mappings for names, locations, and dates.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)

    # Initialize mappings
    all_name_mappings = {}
    all_location_mappings = {}
    all_date_mappings = {}

    # Iterate over all files in the input folder
    for file_name in os.listdir(input_folder_path):
        input_file_path = os.path.join(input_folder_path, file_name)
        
        # Skip directories and non-text files
        if not os.path.isfile(input_file_path) or not file_name.endswith(".txt"):
            continue

        output_file_path = os.path.join(output_folder_path, file_name)

        # Read the input transcript
        with open(input_file_path, "r") as file:
            transcript = file.read()

        # Step 1: Split the transcript into manageable batches
        batches = truncate_transcript(transcript, max_words=1000)

        # Initialize mappings for this file
        name_mapping = {}
        location_mapping = {}
        date_mapping = {}

        # Step 2: Perform de-identification on names, if specified
        if deidentify_name:
            batches, name_mapping = replace_names_with_identifiers(batches, client, model=model)
            print(f"Names de-identified in {file_name}. Mapping: {name_mapping}")

        # Step 3: Perform de-identification on locations, if specified
        if deidentify_location:
            batches, location_mapping = replace_locations_with_identifiers(batches, client, model=model)
            print(f"Locations de-identified in {file_name}. Mapping: {location_mapping}")

        # Step 4: Perform de-identification on dates, if specified
        if deidentify_date:
            batches, date_mapping = replace_dates_with_identifiers(batches, client, model=model)
            print(f"Dates de-identified in {file_name}. Mapping: {date_mapping}")

        # Step 5: Save the de-identified transcript to the output file
        with open(output_file_path, "w") as file:
            for batch in batches:
                file.write(batch + "\n\n")

        print(f"De-identified transcript saved to {output_file_path}")

        # Aggregate mappings if return_mapping is True
        if return_mapping:
            all_name_mappings[file_name] = name_mapping
            all_location_mappings[file_name] = location_mapping
            all_date_mappings[file_name] = date_mapping

    # Step 6: Optionally return the mappings
    if return_mapping:
        return {
            "name_mappings": all_name_mappings,
            "location_mappings": all_location_mappings,
            "date_mappings": all_date_mappings,
        }