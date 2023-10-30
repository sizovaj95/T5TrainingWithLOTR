from bs4 import BeautifulSoup
import re
import requests
import json
import os

import constants as co


title_ignore_indicators = r"awards|filming|production"
question_ignore_indicators = r"film|movie|Tolkien|actor|publish|oscar"
char_map = {
    "’": "'",
    "‘": "'",
    "“": "\"",
    "”": "\"",
    "–": "-",
    "û": "u",
    "é": "e",
    "É": "E",
    "ó": "o",
    "ú": "u",
    "&": "and"
}
QUESTION = "Question"
ANSWER = "Answer"


def _return_soup(url: str) -> BeautifulSoup:
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    return soup


def from_trivia() -> list[dict[str, str]]:
    url = "https://www.trivianerd.com/topic/lord-of-the-rings-trivia#lord-of-the-rings-races-trivia"
    soup = _return_soup(url)
    question_sets = soup.find("article", class_="lg:col-span-9 xl:col-span-6 max-w-prose")
    qa_list = []
    hash_list = []
    for questions in question_sets:
        try:
            title = questions.find("h2").text
            if re.search(title_ignore_indicators, title, re.I):
                continue
            qa_pairs = questions.find(class_="px-4 py-2 space-y-1")
            qa_pairs = qa_pairs.find_all("div")
            for qa in qa_pairs:
                qa_ = [r.text for r in qa.find_all("p")]
                question = qa_[0].replace("Question: ", "")
                question = replace_all_non_compliant_chars(question)
                q_hash = hash(question)
                if q_hash in hash_list:
                    continue
                hash_list.append(q_hash)
                if re.search(question_ignore_indicators, question, re.I):
                    continue
                answer = qa_[1].replace("Answer: ", "")
                answer = replace_all_non_compliant_chars(answer)
                qa_list.append({QUESTION: question, ANSWER: answer})
        except AttributeError:
            continue
    qa_list = remove_non_complete_pairs(qa_list)
    return qa_list


def from_kwizzbit() -> list[dict[str, str]]:
    qa_list = []
    url = "https://kwizzbit.com/lord-of-the-rings-quiz-questions-and-answers/"
    soup = _return_soup(url)
    qa = soup.find_all("ol")
    questions = [q.text for q in qa[0] if q.text.strip()]
    answers = [a.text for a in qa[1] if a.text.strip()]
    numbers = []
    if len(questions) == len(answers):
        for i, question in enumerate(questions):
            if not re.search(question_ignore_indicators, question, re.I):
                numbers.append(i)
        questions = [replace_all_non_compliant_chars(q) for i, q in enumerate(questions) if i in numbers]
        answers = [replace_all_non_compliant_chars(a) for i, a in enumerate(answers) if i in numbers]
    for q, a in zip(questions, answers):
        qa_list.append({QUESTION: q, ANSWER: a})
    qa_list = remove_non_complete_pairs(qa_list)
    return qa_list


def clean_questions() -> list[dict[str, str]]:
    qa_list = []
    with open("../raw_data/questions_unclean.txt", 'r', encoding="utf-16") as f:
        questions = f.read()
    pairs = [qa for qa in questions.split('\n') if len(qa) > 1]
    qa_dict = {}
    skip_answer = False
    for part in pairs:
        if skip_answer:
            skip_answer = False
            continue
        if re.search(question_ignore_indicators, part, re.I):
            skip_answer = True
            continue
        if not part.isascii():
            part = replace_all_non_compliant_chars(part)
        is_q = part.startswith("Question:")
        if is_q:
            qa_dict["Question"] = part.replace("Question:", '').strip()
        else:
            qa_dict["Answer"] = part.replace("Answer:", '').strip()
            qa_list.append(qa_dict)
            qa_dict = {}
    qa_list = remove_non_complete_pairs(qa_list)
    return qa_list


def replace_all_non_compliant_chars(text: str) -> str:
    for incorrect, correct in char_map.items():
        if re.search(incorrect, text):
            text = re.subn(incorrect, correct, text)[0]
    return text


def combine_all_questions():
    whole_set = []
    files = os.scandir(co.QUESTIONS_DATA_FOLDER)
    for file in files:
        if file.name == co.TRAINING_QUESTIONS_FILE_NAME:
            continue
        with open(file.path, "r", encoding="utf-16") as f:
            qa_ = json.load(f)
        whole_set.extend(qa_)
    with open(co.QUESTIONS_DATA_FOLDER / co.TRAINING_QUESTIONS_FILE_NAME, "w") as f:
        json.dump(whole_set, f, indent=4)


def remove_non_complete_pairs(qa_list: list[dict[str, str]]) -> list[dict[str, str]]:
    qa_list = [pair for pair in qa_list if len(pair) == 2]
    return qa_list


def main():
    qa_list = from_trivia()
    with open(co.QUESTIONS_DATA_FOLDER / "trivia.json", "w", encoding="utf-16") as f:
        json.dump(qa_list, f, indent=4)
    qa_list = clean_questions()
    with open(co.QUESTIONS_DATA_FOLDER / "big_quiz.json", "w", encoding="utf-16") as f:
        json.dump(qa_list, f, indent=4)
    qa_list = from_kwizzbit()
    with open(co.QUESTIONS_DATA_FOLDER / "kwizzbit.json", "w", encoding="utf-16") as f:
        json.dump(qa_list, f, indent=4)
    combine_all_questions()


if __name__ == "__main__":
    main()
