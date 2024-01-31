import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import json
import tqdm
import torch
import pickle
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

DEVICE = "cuda"
# MODEL_NAME = "NousResearch/Llama-2-7b-chat-hf"
MODEL_NAME = "NousResearch/Llama-2-13b-hf"
OUTPUT_DIR = "/home/bingxuan/fine-tune/out"

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        use_safetensors=True,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
)
model = PeftModel.from_pretrained(model, OUTPUT_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

DEFAULT_SYSTEM_PROMPT = """
Below is the lyric of a song and the constraint. Rewrite the lyric to satisfy the constraint.
""".strip()

def generate_prompt(input, system_prompt = DEFAULT_SYSTEM_PROMPT):
   return f"""
          ### Instruction: {system_prompt}
   
          ### Input:
          {input}

          ### Response:
          """.strip()

def generate(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    inputs_length = len(inputs["input_ids"][0])
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=1000)
    return tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)

with open('dataset.pkl', 'rb') as handle:
    dataset = pickle.load(handle)

test_dataset = dataset["test"]
test_dataset = test_dataset.map(lambda x: {"prompt": generate_prompt(x["input"])})

save_res = []

for i in tqdm.tqdm(range(len(test_dataset))):
    prompt = test_dataset[i]["prompt"]
    res = generate(model, prompt)
    save_res.append({"prompt": prompt, "res": res})

with open('generated_fine_tuned.json', 'w') as f:
    json.dump(save_res, f)

