from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig
import torch
from torch.cuda.amp import autocast, GradScaler

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


peft_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_rslora=False)

quantization_config = BitsAndBytesConfig(
        # load_in_8bit=True,
        load_in_4bit=True,
        # llm_int8_enable_fp32_cpu_offload=True,
        # llm_int8_has_fp16_weight=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16"
)

device = 'cuda'

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", quantization_config=quantization_config, device_map="auto")

model.config.rope_scaling = {"type": "linear", "factor": 0.3}

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer.pad_token = tokenizer.eos_token

messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)
# model.to(device)

generated_ids = model.generate(model_inputs, do_sample=True, max_new_tokens=10000)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])
