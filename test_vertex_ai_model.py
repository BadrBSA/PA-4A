from google.cloud import aiplatform
import os
from google.oauth2 import service_account

# Chemin vers votre fichier de credentials
credentials_path = "core-site-423419-g8-6f60b4d05c41.json"

# Initialisation des credentials
creds = service_account.Credentials.from_service_account_file(credentials_path)


#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "core-site-423419-g8-6f60b4d05c41.json"

# Initialize Vertex AI
aiplatform.init(project='core-site-423419-g8', location='europe-west1', credentials=creds)

# Define the model ID and endpoint ID
model_id = "projects/498549791590/locations/europe-west1/models/mistralai_mistral-7b-instruct-v0_2-version1"
endpoint_id = "2222003048572518400"

# Create a model instance
model = aiplatform.Model(model_id)

# Create an endpoint instance
endpoint = aiplatform.Endpoint(endpoint_id)


file_path = "HP1.txt"

try:
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
except FileNotFoundError:
    print(f"File {file_path} not found.")
    text = ""


text=f"Harry Potter is a series of fantasy novels written by British author J. K. Rowling. The novels chronicle the lives of a young wizard, Harry Potter, and his friends Hermione Granger and Ron Weasley, all of whom are students at Hogwarts School of Witchcraft and Wizardry. The main story arc concerns Harry's struggle against Lord Voldemort, a dark wizard who intends to become immortal, overthrow the wizard governing body known as the Ministry of Magic, and subjugate all wizards and Muggles (non-magical people)."
instance = {
    "prompt": f"<s>[INST] summarize this text as much as possible: {text} [/INST]",
    "n": 1,
    "max_tokens": 300
}

try:
    # Send prediction request with the prompt input
    response = endpoint.predict(instances=[instance])

    # Process prediction response
    predictions = response.predictions

    # Extract and print the generated summary
    for prediction in predictions:
        prompt_output = prediction.split("Output:\n", 1)
        if len(prompt_output) > 1:
            generated_text = prompt_output[1].strip()
            print("text input length: ", len(text))
            print("text input: ", text)
            print("Generated summary length: ", len(generated_text))
            print("Generated summary:", generated_text)
        else:
            print("Unable to extract the generated summary from the response.")

except Exception as e:
    # Log the error message
    print(f"An error occurred during prediction: {e}")

    # Check if more details can be obtained from the error
    if hasattr(e, 'details'):
        print(f"Error details: {e.details}")