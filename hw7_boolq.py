import os
import openai
import torch
from datasets import load_dataset
from datasets.combine import concatenate_datasets
import evaluate as evaluate
from tqdm import tqdm
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_prompts(train_size, test_size):
    dataset = load_dataset("boolq")
    dataset = dataset.shuffle()
    
    train_dataset_true = dataset["train"].filter(lambda example: example["answer"] == True).select(range(int(train_size / 2)))
    train_dataset_false = dataset["train"].filter(lambda example: example["answer"] == False).select(range(train_size - int(train_size / 2)))
    
    train_dataset = concatenate_datasets([train_dataset_true, train_dataset_false])
    train_dataset = train_dataset.shuffle()
    
    test_dataset = dataset["validation"].select(range(test_size))
    
    prompt = [f'Question: {example["question"]}\nPassage: {example["passage"]}\nAnswer: {example["answer"]}\n' for example in train_dataset]
    prompt = "\n".join(prompt)
    
    test_prompts = [f'\nQuestion: {example["question"]}\nPassage: {example["passage"]}\nAnswer: ' for example in test_dataset]
    test_answers = [example["answer"] for example in test_dataset]
    
    return prompt, test_prompts, test_answers

def response_bloomz(train_prompt, test_prompts, device):
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-1b1")
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-1b1", torch_dtype="auto", device_map="auto")
    model.to(device)
    model.eval()
    responses = []
    for test_prompt in tqdm(test_prompts):
        prompt_data = train_prompt + "\n" + test_prompt
        inputs = tokenizer.encode(prompt_data, return_tensors="pt").to(device)
        outputs = model.generate(inputs, max_length=5000)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Question: ")[-1]
        print(response)
        result = 1 if response == " True" else 0
        responses.append(result)
    return responses

def response_openai(train_prompt, test_prompts, model):
    openai.api_key = "OPENAI_API"
    responses = []
    for test_prompt in tqdm(test_prompts):
        response = openai.Completion.create(
            model=model,
            prompt=train_prompt + "\n" + test_prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        print(response)
        result = 1 if response.choices[0].text == " True" else 0
        responses.append(result)
    return responses

def get_accuracy(train_prompt, test_prompts, test_answers, model, device):
    if model == "bloomz":
        responses = response_bloomz(train_prompt, test_prompts, device)
    else:
        responses = response_openai(train_prompt, test_prompts, model)
    accuracy = evaluate.load("accuracy")
    for response, answer in zip(responses, test_answers):
        accuracy.add(predictions=response, references=answer)
    return accuracy.compute()

if __name__ == "__main__":
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="text-davinci-003")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_size", type=int, default=8)
    parser.add_argument("--test_size", type=int, default=30)
    
    args = parser.parse_args()
    train_prompt, test_prompts, answers = generate_prompts(args.train_size, args.test_size)
    
    accuracy = get_accuracy(train_prompt, test_prompts, answers, args.model, args.device)

    print(accuracy)
