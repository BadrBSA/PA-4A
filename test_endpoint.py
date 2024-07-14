import requests

url = "https://7456-2a01-e0a-a5b-5df0-00-4fec-8881.ngrok-free.app/summarize"  # Replace with your server's IP address

with open("data/books/first_chapter", "r") as file:
    text = file.read()

data = {"text": text}
response = requests.post(url, json=data)
summary = response.json()["summary"]


print(summary)
