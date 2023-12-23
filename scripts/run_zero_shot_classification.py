import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import evaluate

from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPTS = [
    "{text} => Sarcasm:",
    "Text: {text} => Sarcasm:",
    "{text}\nIs this text above sarcastic or not?",
    "Is the following text sarcastic?\nText: {text}\nAnswer:",
    "Text: {text}\nPlease classify the text above for sarcasm.",
]


@torch.no_grad()
def get_logprobs(model, tokenizer, prompt, device, max_length):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    input_ids, output_ids = inputs["input_ids"], inputs["input_ids"][:, 1:]

    outputs = model(**inputs, labels=input_ids)
    logits = outputs.logits

    logprobs = torch.gather(F.log_softmax(logits, dim=2), 2, output_ids.unsqueeze(2))
    return logprobs.sum()


def predict_classification(model, tokenizer, input_text, labels, device, max_length):
    probs = [get_logprobs(model, tokenizer, f"{input_text} {label}", device, max_length) for label in labels]
    return probs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--model_max_length", type=int, default=1024)
    parser.add_argument("--load_8bit", action="store_true")
    parser.add_argument("--dtype", default="float16", type=str)
    parser.add_argument("--output_folder", type=str, default="outputs")

    return parser.parse_args()


def main():
    args = parse_args()
    output_folder = f"{args.output_folder}/{args.base_model.split('/')[-1]}"
    os.makedirs(output_folder, exist_ok=True)

    # load dataset, tokenizer, model
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    dataset_label_list = ["not sarcastic", "sarcastic"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=args.load_8bit,
        torch_dtype=getattr(torch, args.dtype),
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()

    # metrics
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")

    def compute_metrics(predictions, labels):
        return {
            "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
            "f1": f1.compute(predictions=predictions, references=labels)["f1"],
            "precision": precision.compute(predictions=predictions, references=labels)["precision"],
            "recall": recall.compute(predictions=predictions, references=labels)["recall"],
        }

    metrics = {}
    for prompt_id, prompt in enumerate(PROMPTS):
        labels, predictions = [], []
        for datum in tqdm(dataset):
            # apply input prompt
            input_text = prompt.format(**datum)

            # forward pass
            probs = predict_classification(
                model, tokenizer, input_text, dataset_label_list, device, args.model_max_length
            )
            pred_idx = np.argmax([prob.cpu().detach() for prob in probs])

            # collect label and prediction
            labels.append(datum["label"])
            predictions.append(pred_idx)

        metrics[prompt_id] = compute_metrics(predictions, labels)

    # average metrics across prompts
    mean_metrics = {}
    for metric_name in metrics[0]:
        mean_metrics[metric_name] = sum(m[metric_name] for m in metrics.values()) / len(metrics)
    metrics["mean"] = mean_metrics

    with open(f"{output_folder}/eval_results_{args.dataset_name.split('/')[-1]}.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
