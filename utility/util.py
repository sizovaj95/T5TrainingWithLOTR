import math
import random
import re
import numpy as np
from datetime import datetime as dt
import torch
from pathlib import Path

noise_density = 0.15
mean_noise_span_length = 3


char_map = {
    "’": "'",
    "‘": "'",
    "“": "\"",
    "”": "\"",
    "–": "-",
    "û": "u",
    "é": "e",
    "É": "E",
    "È": "E",
    "ó": "o",
    "ú": "u",
    "&": "and",
    "Ì": "I",
    "Í": "I",
    "Ó": "O",
    "Ú": "U",
    "á": "a",
    "í": "i",
    "ä": "a"
}


def apply_sentinel_tokens(text: str, mask: list[int], to_mask: int) -> str:
    to_replace = []
    i = 0
    pairs = list(zip(text.split(), mask))
    pairs.append(('dummy', None))
    for word, mask_ in pairs:
        if mask_ == to_mask:
            to_replace.append(word)
        elif to_replace:
            to_replace_text = ' '.join(to_replace)
            text = re.sub(re.escape(to_replace_text), f"<extra_id_{i}>", text)
            to_replace = []
            i += 1
    return text


def apply_mask_to_input_text(text: str) -> tuple[str, str]:
    min_words_required = math.ceil(1.5/noise_density)
    words = text.split()
    text_len = len(words)
    # if text_len < min_words_required:
    #     raise ValueError(f"Input text '{text}' is too short!")

    num_words_to_drop = math.floor((text_len * noise_density) / 1.5)
    mask_indices = random_spans_noise_mask(text_len, num_words_to_drop).astype(int)
    input_text = return_masked_text(mask_indices, words)
    output_text = return_masked_text(1 - mask_indices, words)

    return input_text, output_text


def return_masked_text(mask_indices: np.ndarray, words: list[str]) -> str:
    start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
    sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
    sentinel_ids -= mask_indices - start_indices

    masked_input = np.where(sentinel_ids != 0, sentinel_ids, words)
    masked_input = [i for i in masked_input if i != '-1']
    for i, item in enumerate(masked_input):
        try:
            index = int(item) - 1
            masked_input[i] = f"extra_id_{index}"
        except ValueError:
            continue
    return ' '.join(masked_input)


def random_segmentation(num_items: int, num_segments: int) -> np.ndarray:
    """Partition a sequence of items randomly into non-empty segments.
    Args:
        num_items: an integer scalar > 0
        num_segments: an integer scalar in [1, num_items]
    Returns:
        a Tensor with shape [num_segments] containing positive integers that add
        up to num_items
    """
    mask_indices = np.arange(num_items - 1) < (num_segments - 1)
    np.random.shuffle(mask_indices)
    first_in_segment = np.pad(mask_indices, [[1, 0]])
    segment_id = np.cumsum(first_in_segment)
    # count length of sub segments assuming that list is sorted
    _, segment_length = np.unique(segment_id, return_counts=True)
    return segment_length


def random_spans_noise_mask(length: int, num_noise_tokens=None) -> np.ndarray:
    """This function is copy of `random_spans_helper
    <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .

    Noise mask consisting of random spans of noise tokens.
    The number of noise tokens and the number of noise spans and non-noise spans
    are determined deterministically as follows:
    num_noise_tokens = round(length * noise_density)
    num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
    Spans alternate between non-noise and noise, beginning with non-noise.
    Subject to the above restrictions, all masks are equally likely.

    Args:
        length: an int32 scalar (length of the incoming token sequence)
        noise_density: a float - approximate density of output mask
        mean_noise_span_length: a number

    Returns:
        a boolean tensor with shape [length]
    """

    orig_length = length

    if not num_noise_tokens:
        num_noise_tokens = int(np.round(length * noise_density))
    num_nonnoise_tokens = length - num_noise_tokens
    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    # num_noise_tokens should be less than num_noise_tokens and num_nonnoise_tokens
    num_noise_spans = int(np.round(min(num_noise_tokens, num_nonnoise_tokens) / mean_noise_span_length))

    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = max(num_noise_spans, 1)

    noise_span_lengths = random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = random_segmentation(num_nonnoise_tokens, num_noise_spans)

    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
    )
    span_starts = np.cumsum(interleaved_span_lengths)[:-1]
    span_start_indicator = np.zeros((length,), dtype=np.int8)
    try:
        span_start_indicator[span_starts] = True
    except IndexError:
        span_start_indicator
    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)

    return is_noise[:orig_length]


def save_model(model, model_name: str):
    save_folder = Path(__file__).parent.parent.resolve() / "models"
    try:
        print(f"Saving {model_name}")
        torch.save(model, str(save_folder) + '//' + model_name)
    except Exception as er:
        print(f"Failed to save {model_name}: {er}")


def replace_all_non_compliant_chars(text: str) -> str:
    for incorrect, correct in char_map.items():
        if re.search(incorrect, text):
            text = re.subn(incorrect, correct, text)[0]
    return text