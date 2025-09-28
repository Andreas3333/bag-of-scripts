#!/usr/bin/env -S uv run --script --env-file .env
#
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "torch",
#   "transformers"
# ]
# ///

"""Generative HuggingFace Model Chat, load and prompt"""


import os
from datetime import datetime
import secrets
import hashlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  ## Causal model for chat/Instruct


class Colors:
    chat = '\033[94m'
    info = '\033[95m'
    RESET = '\033[0m'


def print_info(message: str):
    """prints message with info color"""
    print("-"*90)
    print(f"{Colors.info}{message}{Colors.RESET}")
    print("-"*90, "\n")

def print_msg(message: str):
    """prints message with chat color"""
    print(f"\n{Colors.chat}{message}{Colors.RESET}\n")

def gen_short_hash(length=8):
    random_bytes = secrets.token_bytes(32)
    sha256_hash = hashlib.sha256(random_bytes).hexdigest()
    return sha256_hash[:length]

def get_date():
    now = datetime.now()
    year = now.year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    second = now.second

    return f"{year}-{month:02d}-{day:02d}-{hour:02d}:{minute:02d}:{second:02d}"

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.backends.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def create_conversation_file():
    conversation_dir = f"conversations/{checkpoint}"
    os.makedirs(conversation_dir, exist_ok=True)
    conversation = get_date() + "-" + gen_short_hash() + '.md'
    conversation_file = f"{conversation_dir}/{conversation}"
    open(conversation_file, 'x').close()
    return conversation_file

def append_to_conversation(convo_file: str, prompt, content):
    with open(convo_file, 'a') as f:
        f.write(f"# (Prompt):\n\n{prompt}\n\n## ({checkpoint} output):\n\n{content}\n")


checkpoint = os.getenv("HF_MODEL")
save_conversations = os.getenv("SAVE_CONVERSATIONS", False)
device = get_device()

tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_safetensors=True, local_files_only=True, device=device)
model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True, use_safetensors=True)
model.to(device)


def main():

    if save_conversations:
        convo_file = create_conversation_file()

    print_info(f"You are chatting with the {checkpoint} model.\
            \nYour conversations { f"will be saved at conversations/{checkpoint}/" if save_conversations else "will not be saved" }.\
            \nPress 'q' to quit")

    prompt = input(f"{Colors.chat}Input prompt >>>{Colors.RESET}\n")

    while prompt != 'q':
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=16384
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        content = tokenizer.decode(output_ids, skip_special_tokens=True)

        print(f"\n{Colors.chat}{checkpoint} output:{Colors.RESET}\n", content)
        if save_conversations:
            append_to_conversation(convo_file, prompt, content)
        prompt = input(f"{Colors.chat}Input prompt >>>{Colors.RESET}\n")

    print_info("Exiting...")

if __name__ == "__main__":
    main()
