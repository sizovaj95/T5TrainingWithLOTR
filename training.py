import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
from tqdm import tqdm
import time
from datetime import datetime as dt
from sklearn.model_selection import train_test_split

from utility import util

EPOCHS = 4
MODEL_NAME = "t5-base"
BATCH_SIZE = 8
MAX_OUTPUT_LEN = 512
MAX_INPUT_LEN = 800
TRAIN_FRACTION = 0.8
RANDOM_SEED = 42


def train(epochs: int, model, device, train_loader: DataLoader, optimizer, pad_token_id: int):
    model.train()

    for epoch in range(epochs):
        for train_data in tqdm(train_loader):

            input_ids = train_data['input_ids'].to(device)
            input_mask = train_data['input_mask'].to(device)

            target_ids = train_data['output_ids'].to(device)
            target_ids[target_ids == pad_token_id] = -100

            outputs = model(input_ids=input_ids, attention_mask=input_mask, labels=target_ids)
            loss = outputs[0]
            print(f"Epoch {epoch+1} | Loss: {loss.item()}")

            optimizer.zero_grad()
            try:
                loss.backward()
            except Exception as ex:
                print(ex)
                raise
            optimizer.step()


def validate(model: T5ForConditionalGeneration, test_loader: DataLoader, tokenizer: T5Tokenizer, device):
    model.eval()

    predictions = []
    targets = []
    with torch.no_grad():
        for test_data in test_loader:
            target_ids = test_data['output_ids'].to(device)
            input_ids = test_data['input_ids'].to(device)
            input_mask = test_data['input_mask'].to(device)

            predicted_ids = model.generate(
                input_ids=input_ids,
                attention_mask=input_mask,
                max_length=MAX_OUTPUT_LEN
            )
            predictions_ = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                            for g in predicted_ids]
            targets_ = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in
                        target_ids]

            predictions.extend(predictions_)
            targets.extend(targets_)
    return predictions, targets


def get_max_token_numbers_from_training_data():
    # with open("training_data.json", "r", encoding="utf8") as f:
    #     data = json.load(f)
    # df = pd.DataFrame.from_records(data)
    df = pd.read_csv("lotr_train_data.csv")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    review_lens = []
    summary_lens = []
    for i, row in df.iterrows():
        text = row['text']
        # summary = row['summary']
        rev_len = len(tokenizer.encode(text, max_length=5000))
        # sum_len = len(tokenizer.encode(summary, max_length=500))
        review_lens.append(rev_len)
        # summary_lens.append(sum_len)
    rev_max = max(review_lens)
    # sum_max = max(summary_lens)

    print(f"Longest review: {rev_max} tokens")
    # print(f"Longest summary: {sum_max} tokens")
    print("======================================")


def save_model(model, train_df_len: int):
    today = dt.now()
    dt_now_str = f"{today.year}{today.month}{today.day}"
    model_name = f"{train_df_len}_{dt_now_str}_{MAX_OUTPUT_LEN}_{EPOCHS}.pt"
    try:
        print(f"Saving {model_name}")
        torch.save(model, "models//" + model_name)
    except Exception as er:
        print(f"Failed to save {model_name}: {er}")


def print_test_results(model: T5ForConditionalGeneration, tokenizer: T5Tokenizer, test_data, device):
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    predictions, targets = validate(model, test_loader, tokenizer, device)
    # scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    # for pred, tar in zip(predictions, targets):
    #     print(f"Target: {tar}\nPredicted: {pred}")
    #     scores = scorer.score(tar, pred)
    #     print(scores)
    #     print("\n")


def load_and_split_data() -> tuple:
    data = pd.read_csv("training_data/lotr_train_data.csv")
    train_df, test_df = train_test_split(data, train_size=TRAIN_FRACTION, random_state=RANDOM_SEED)
    return train_df, test_df


class CustomDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer, max_input_len: int, max_output_len: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        text = self.data.iloc[idx]['text']
        try:
            masked_input, masked_output = util.apply_mask_to_input_text(text)
        except ValueError as err:
            print(err)
            return None

        encoded_input = self.tokenizer(masked_input, max_length=self.max_input_len, pad_to_max_length=True,
                                       truncation=True,
                                       padding="max_length", return_tensors="pt")
        encoded_output = self.tokenizer(masked_output, max_length=self.max_output_len, pad_to_max_length=True,
                                        truncation=True,
                                        padding="max_length", return_tensors="pt")

        return {
            "context": text,
            "input_ids": encoded_input['input_ids'].squeeze(),
            "input_mask": encoded_input['attention_mask'].squeeze(),
            "output_ids": encoded_output['input_ids'].squeeze()
        }


def collate_fn(batch):
    batch = [item for item in batch if item is not None]

    return batch


def main():
    torch.manual_seed(RANDOM_SEED)
    # model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, model_max_length=MAX_INPUT_LEN)
    device = torch.device('cuda')

    lr = 1e-4

    train_df, test_df = load_and_split_data()
    train_df = train_df[train_df['text'].str.split().str.len() >= 5]

    # get_max_token_numbers_from_training_data()

    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    # for param in model.base_model.parameters():
    #     param.requires_grad = False
    for name, param in model.base_model.encoder.named_parameters():
        param.requires_grad = False

    train_dataset = CustomDataset(train_df, tokenizer, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
    # train_dataset[3]
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)#, collate_fn=collate_fn)
    # train_loader

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train(EPOCHS, model, device, train_loader, optimizer, tokenizer.pad_token_id)
    save_model(model, len(train_df))


def test_trained_model():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, model_max_length=MAX_INPUT_LEN)
    model_bare = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model_frozen = torch.load("models/7106_20231016_512_1.pt").to(torch.device('cpu'))
    model_frozen.eval()
    model_bare.eval()

    # model_unfrozen = torch.load("models/20_2023714_512_3.pt")
    # model_unfrozen.eval()

    # text = 'question: ' + 'who is Theoden?'
    text = 'question: ' + "Where do the hobbits live?"
    # text = "The house that was built in <extra_id_0>, although old but still <extra_id_1> and cosy."
    # text = "Theoden is <extra_id_0> Rohan."
    # text = "The <extra_id_0> walks in <extra_id_1> park"
    # text = "This hobbit was a very <extra_id_0> and his name was Baggins."
    # text = "In a hole in the <extra_id_0> a hobbit. Not a nasty, dirty, <extra_id_1> with the ends " \
    #            "of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or " \
    #            "<extra_id_2> it was a hobbit'hole, and that means comfort"
    encoded = tokenizer(text, max_length=MAX_INPUT_LEN, pad_to_max_length=True,
                        # truncation=True,
                        # padding="max_length",
                        return_tensors="pt",
                        add_special_tokens=True)
    pred_ids = model_frozen.generate(
        input_ids=encoded.input_ids,
        # attention_mask=encoded.attention_mask,
        # max_length=30,  # Set a maximum length for the generated text
        min_length=20,  # Set a minimum length for the generated text
        no_repeat_ngram_size=2,  # Avoid repeated n-grams
        top_k=50,  # Consider the top 50 most likely tokens at each step
        top_p=0.95  # Consider tokens until the cumulative probability exceeds 95%


    )
    pred_text_tr = tokenizer.decode(pred_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    sequences_frozen = tokenizer.batch_decode(pred_ids)

    # pred_ids = model_unfrozen.generate(
    #     input_ids=encoded.input_ids,
    #     # attention_mask=encoded.attention_mask,
    #     # max_length=MAX_OUTPUT_LEN
    # )
    # pred_text_tr = tokenizer.decode(pred_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # sequences_unfrozen = tokenizer.batch_decode(pred_ids)

    pred_ids = model_bare.generate(
        input_ids=encoded.input_ids,
        # attention_mask=encoded.attention_mask,
        # max_length=30,  # Set a maximum length for the generated text
        min_length=20,  # Set a minimum length for the generated text
        no_repeat_ngram_size=2,  # Avoid repeated n-grams
        top_k=50,  # Consider the top 50 most likely tokens at each step
        top_p=0.95  # Consider tokens until the cumulative probability exceeds 95%

    )
    sequences_bare = tokenizer.batch_decode(pred_ids)
    pred_text_br = tokenizer.decode(pred_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return


if __name__ == "__main__":
    start = time.time()
    main()
    # get_max_token_numbers_from_training_data()
    # test_trained_model()
    print("Time taken: {0:.2f} s".format(time.time() - start))
