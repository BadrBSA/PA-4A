import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datetime import datetime
import os


def enlever_inst(contenu):
    # Trouver la position de la balise [/INST]
    fin_inst = contenu.find("[/INST]")

    # Vérifier si la balise [/INST] a été trouvée
    if fin_inst != -1:
        # Supprimer tout le texte avant et y compris la balise [/INST]
        resultat = contenu[fin_inst + len("[/INST]") :]
        return resultat
    else:
        # Si la balise [/INST] n'a pas été trouvée, retourner le contenu original
        return contenu


def split_text_into_token_segments(text, tokenizer, token_amount=5000):
    segments = []

    encoded_text = tokenizer.encode(text=text, return_tensors="pt")

    sequence = []
    for tensor in encoded_text[0]:
        sequence.append(tensor)

        if len(sequence) == token_amount:
            segments.append(sequence)
            sequence = []

    if sequence:
        segments.append(sequence)

    return segments


def summarize_token_segment(
    segments,
    model,
    tokenizer,
    device,
    instructions="make me a summary of the following text: ",
):

    summaries = []
    for segment in segments:
        text_to_summarize = tokenizer.decode(segment, skip_special_tokens=True)
        message = [
            {"role": "user", "content": f"{instructions}" f"{text_to_summarize}"}
        ]

        encodeds = tokenizer.apply_chat_template(message, return_tensors="pt").to(
            "cuda"
        )

        generated_ids = model.generate(
            encodeds,
            do_sample=True,
            max_new_tokens=1000,
            pad_token_id=tokenizer.pad_token_id,
        )
        decoded = tokenizer.batch_decode(generated_ids)

        summaries.append([enlever_inst(decoded[0])])

    return summaries


def group_summaries(summaries, tokenizer):
    grouped_summaries = "".join([item for sublist in summaries for item in sublist])
    return split_text_into_token_segments(grouped_summaries, tokenizer)


def full_process(text, model, tokenizer, device):
    token_segments = split_text_into_token_segments(text, tokenizer)
    while True:
        summary = summarize_token_segment(
            token_segments,
            model,
            tokenizer,
            device,
            "Your job is to summarize very long texts. Your task is to generate an appropriate summary based on the following given text.",
        )
        grouped_summaries = group_summaries(summary, tokenizer)
        if len(grouped_summaries) == 1:
            return summarize_token_segment(
                grouped_summaries,
                model,
                tokenizer,
                "Your job is to summarize very long texts. Your task is to generate an appropriate summary based on the following given text.",
            )
        else:
            token_segments = grouped_summaries


if __name__ == "__main__":
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

    with open("data/books/first_chapter", "r", encoding="utf-8") as f:
        text = f.read()

    summary = full_process(text)

    file_name = f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    directory = "data/books/"
    file_path = os.path.join(directory, file_name)

    with open(file_path, "w") as file:
        file.write(summary[0][0])
