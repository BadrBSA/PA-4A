from flask import Flask, request, jsonify
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
from generate_summary import full_process


app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16"
)

base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    quantization_config=quantization_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, "data/model/first_model").to(device)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer.pad_token = tokenizer.eos_token


@app.route("/")
def index():
    return {"Hello": "World"}


@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.json
    text = data["text"]

    summary = full_process(text, model, tokenizer, device)

    return jsonify({"summary": summary[0][0]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
