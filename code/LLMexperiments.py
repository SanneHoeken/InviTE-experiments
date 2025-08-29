import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import os


models = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen2-7B-Instruct"
]

prompt_files = ["../data/prompts/prompt1.txt"]

input_file = "...csv"
output_dir = "../output/"
os.makedirs(output_dir, exist_ok=True)


def build_prompt(template, sentence):
    return template.replace("[SENTENCE]", sentence).strip()


def clean_response(response_text, prompt):
    text = response_text.replace(prompt, "").replace("\n", "").strip().split("###")[0]
    if text.lower().startswith("invective"):
        return 1
    elif text.lower().startswith("non-invective"):
        return 0
    return text


def classify_sentence_chat(sentence, prompt_template, tokenizer, model):
    prompt = build_prompt(prompt_template, sentence)
    messages = [
        {"role": "system", "content": "You are an expert in historical linguistics analyzing invective language in Tudor England texts."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=20,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]

    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
     
    return response_text


def classify_sentence(sentence, prompt_template, tokenizer, pipe):
    
    prompt = build_prompt(prompt_template, sentence)
    response = pipe(
        prompt, 
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=1, 
        max_new_tokens=20)
        
    return clean_response(response[0]['generated_text'], prompt)



def main():

    df = pd.read_csv(input_file, encoding='unicode_escape', sep=';')

    for model_name in models:
        print(f"\nLoading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        #model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
        pipe = pipeline("text-generation", model=model_name, device_map="auto") 

        for prompt_path in prompt_files:
            prompt_name = os.path.splitext(os.path.basename(prompt_path))[0]
            print(f"Using prompt: {prompt_name}")

            with open(prompt_path, "r") as pf:
                prompt_template = pf.read()

            tqdm.pandas(desc=f"{model_name.split('/')[-1]} | {prompt_name}")
            df[f'pred_{model_name.split("/")[-1]}_{prompt_name}'] = df['originalsentence'].progress_apply(
                #lambda s: classify_sentence_chat(s, prompt_template, tokenizer, model))
                lambda s: classify_sentence(s, prompt_template, tokenizer, pipe))
            

            output_file = os.path.join(
                output_dir,
                f"{model_name.split('/')[-1]}_{prompt_name}_preds.csv")
            df.to_csv(output_file, index=False)
            print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
