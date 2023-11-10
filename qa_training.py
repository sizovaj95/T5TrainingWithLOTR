from pathlib import Path
import json
import time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
from tqdm import tqdm
from datetime import datetime as dt
from sklearn.model_selection import train_test_split

from utility import util

MODEL_NAME = "t5-base"
BATCH_SIZE = 4
EPOCHS = 100


class QaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: T5Tokenizer, q_len: int, a_len: int):
        self.df = df
        self.tokenizer = tokenizer
        self.q_len = q_len
        self.a_len = a_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        question = row['Question']
        answer = row['Answer']
        question_tokenized = self.tokenizer(question, max_length=self.q_len, padding="max_length",
                                            truncation=True, pad_to_max_length=True, add_special_tokens=True,
                                            return_tensors="pt")
        answer_tokenized = self.tokenizer(answer, max_length=self.a_len, padding="max_length",
                                          truncation=True, pad_to_max_length=True, add_special_tokens=True,
                                          return_tensors="pt")

        labels = answer_tokenized["input_ids"].squeeze()
        labels[labels == 0] = -100
        return {
            "input_ids": question_tokenized["input_ids"].squeeze(),
            "attention_mask": question_tokenized["attention_mask"].squeeze(),
            "labels": labels,
            "decoder_attention_mask": answer_tokenized["attention_mask"].squeeze()
        }


def load_data() -> pd.DataFrame:
    folder = Path(__file__).parent.resolve() / 'training_data' / 'questions'
    with open(folder / "training_questions.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    data = pd.DataFrame(data)
    return data


def main():
    lr = 1e-4
    df = load_data()
    df['q_len'] = df['Question'].apply(lambda x: len(x))
    df['a_len'] = df['Answer'].apply(lambda x: len(x))

    max_q_len = df.q_len.max()
    max_a_len = df.a_len.max()

    device = torch.device("cuda")

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, model_max_length=max_q_len)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

    train_dataset = QaDataset(df, tokenizer, max_q_len, max_a_len)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loss = 0
    train_batch_count = 0
    for epoch in range(EPOCHS):
        model.train()
        for batch in tqdm(train_loader, desc="Training batches"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            decoder_attention_mask = batch["decoder_attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask
            )

            optimizer.zero_grad()
            outputs.loss.backward()
            optimizer.step()
            train_loss += outputs.loss.item()
            train_batch_count += 1
        print(
            f"{epoch + 1}/{2} -> Train loss: {train_loss / train_batch_count}")

    today = dt.now()
    dt_now_str = f"{today.year}{today.month}{today.day}"
    model_name = f"{len(df)}_{dt_now_str}_{max_a_len}_{EPOCHS}.pt"
    util.save_model(model, model_name)


def try_saved_model():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    save_folder = Path(__file__).parent.resolve() / "models"
    model = torch.load(str(save_folder) + '//' + '134_2023113_79_100.pt').to(torch.device('cpu'))
    model.eval()

    question = "Who wrote The Lord of the Rings?"
    encoded = tokenizer(question, pad_to_max_length=True,
                        # truncation=True,
                        # padding="max_length",
                        return_tensors="pt",
                        add_special_tokens=True)
    pred_ids = model.generate(
        input_ids=encoded.input_ids,
        # attention_mask=encoded.attention_mask,
        # max_length=30,  # Set a maximum length for the generated text
        min_length=20,  # Set a minimum length for the generated text
        no_repeat_ngram_size=2,  # Avoid repeated n-grams
        top_k=50,  # Consider the top 50 most likely tokens at each step
        top_p=0.95)  # Consider tokens until the cumulative probability exceeds 95%

    pred_text_tr = tokenizer.decode(pred_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    sequences_frozen = tokenizer.batch_decode(pred_ids)
    return


if __name__ == "__main__":
    try_saved_model()

