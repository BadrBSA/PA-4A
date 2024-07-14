import os
import warnings
import pandas as pd

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

from sklearn.model_selection import train_test_split


# Fonction pour générer un prompt d'entraînement
def generate_prompt(datapoint):
    return f"""
            The following data comes from a dataset made of chapters and their summaries
            [INST] Your job is to summarize very long texts. Your task is to generate an appropriate summary based on the text given in square brackets.
            [{datapoint['Context']}][/INST]

            {datapoint['Response']}""".strip()


# Fonction pour générer un prompt de test
def generate_test_prompt(datapoint):
    return f"""
            The following data comes from a dataset made of chapters and their summaries
            [INST] Your job is to summarize very long texts. Your task is to generate an appropriate summary based on the text given in square brackets.
            [{datapoint['Context']}][/INST]""".strip()


os.environ["WANDB_DISABLED"] = "true"

# Configuration des variables d'environnement pour l'utilisation du GPU et des tokenizers
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")

# Charger le dataset "booksum-short"
dataset = load_dataset("pszemraj/booksum-short")

# Extraire les données d'entraînement du dataset
train_data = dataset["train"]
chapters = train_data["chapter"]
summaries = train_data["summary_text"]

# Créer un DataFrame pandas à partir des chapitres et des résumés
df = pd.DataFrame({"Context": chapters, "Response": summaries})

# Diviser les données en ensembles d'entraînement et de validation
X_train, X_eval = train_test_split(df, test_size=0.2, random_state=42)

# Générer les prompts pour l'entraînement et la validation
X_train = pd.DataFrame(X_train.apply(generate_prompt, axis=1), columns=["text"])
X_eval = pd.DataFrame(X_eval.apply(generate_test_prompt, axis=1), columns=["text"])

# Convertir les DataFrames pandas en datasets
train_data = Dataset.from_pandas(X_train)
eval_data = Dataset.from_pandas(X_eval)

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

compute_dtype = getattr(torch, "float16")

# Configuration de la quantization en 4 bits
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

# Charger le modèle préentraîné avec la configuration de quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
)

# Désactiver le cache pour le modèle
model.config.use_cache = False
model.config.pretraining_tp = 1

# Charger le tokenizer associé au modèle
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    add_eos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)

run_name = "mistral-7b-instruct-PA4A"
output_dir = "./" + run_name

# Configuration LoRA
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)

# Arguments d'entraînement
training_arguments = TrainingArguments(
    output_dir=output_dir,
    logging_dir="logs",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_strategy="epoch",
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.05,
    group_by_length=True,
    lr_scheduler_type="cosine",
    evaluation_strategy="epoch",
    do_eval=True,
    run_name=run_name,
    disable_tqdm=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
    max_seq_length=512,
)

# Appliquer la configuration LoRA au modèle
model = get_peft_model(model, peft_config)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
torch.cuda.empty_cache()

trainer.train()

model.save_pretrained("data/model/first_model")
trainer.save_model("data/model/first_model_trainer")
