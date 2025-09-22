#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于Transformers的生成式金融聊天：
- chat: 直接加载预训练模型进行对话
- train(optional): 使用 Finance-Instruct-500k 做轻量微调
"""

import os
import argparse
from typing import List, Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


def prepare_dataset(tokenizer, max_length: int = 1024, split: str = "train"):
    dataset = load_dataset("Josephgflowers/Finance-Instruct-500k")
    data = dataset[split]

    def format_example(example: Dict[str, Any]) -> Dict[str, str]:
        # 构造简单的指令式样本
        user = example.get("user", "").strip()
        assistant = example.get("assistant", "").strip()
        text = f"User: {user}\nAssistant: {assistant}"
        return {"text": text}

    data = data.map(format_example, remove_columns=data.column_names)

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
        )

    tokenized = data.map(tokenize, batched=True, remove_columns=["text"]) 
    return tokenized


def train(model_name: str, output_dir: str, num_train_steps: int = 1000, lr: float = 5e-5, batch_size: int = 2):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    train_ds = prepare_dataset(tokenizer, split="train")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=50,
        save_steps=200,
        max_steps=num_train_steps,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=1,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


class GenerativeChat:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.eval()

    @torch.no_grad()
    def generate(self, history: List[Dict[str, str]], user_input: str, max_new_tokens: int = 128) -> str:
        # 简单拼接历史
        context = " ".join([f"User: {h['user']} Assistant: {h['bot']}" for h in history[-3:]]) if history else ""
        prompt = f"{context} User: {user_input} Assistant:"

        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        resp = text[len(prompt):].strip()
        return resp


def run_chat(model_path: str):
    chat = GenerativeChat(model_path)
    history: List[Dict[str, str]] = []
    print("=" * 60)
    print("Generative Financial Chat (Transformers)")
    print("Type 'exit' to quit.")
    print("=" * 60)
    while True:
        try:
            user = input("\nYou: ").strip()
            if not user:
                continue
            if user.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            reply = chat.generate(history, user)
            print(f"Assistant: {reply}")
            history.append({"user": user, "bot": reply})
        except KeyboardInterrupt:
            print("\nInterrupted. Bye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "chat"], help="train or chat")
    parser.add_argument("--model_name", default="microsoft/DialoGPT-medium")
    parser.add_argument("--output_dir", default="models/generative-finetune")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    if args.mode == "train":
        train(
            model_name=args.model_name,
            output_dir=args.output_dir,
            num_train_steps=args.steps,
            lr=args.lr,
            batch_size=args.batch_size,
        )
    else:
        # 推理：可使用微调产物目录或任意兼容CausalLM模型名称
        model_path = args.output_dir if os.path.exists(args.output_dir) else args.model_name
        run_chat(model_path)


if __name__ == "__main__":
    main()


