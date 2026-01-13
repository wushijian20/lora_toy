import re
from pathlib import Path

def parse_chat_file(path):
    conversations = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Match speaker lines: *CHI:, *EXP:, etc.
            if re.match(r"^\*[A-Z]{3}:", line):
                speaker, text = line.split(":", 1)

                # Basic cleaning
                text = text.strip()
                text = re.sub(r"\[.*?\]", "", text)   # remove CHAT annotations
                text = re.sub(r"&[a-z]+", "", text)   # remove &-um, &oh
                text = re.sub(r"\s+", " ", text)

                if text:
                    conversations.append({
                        "speaker": speaker[1:],  # CHI, EXP
                        "text": text
                    })

    return conversations


data_dir = Path("chat_files")
all_samples = []

for chat_file in data_dir.glob("*.cha"):
    samples = parse_chat_file(chat_file)
    for s in samples:
        s["source_file"] = chat_file.name
        all_samples.append(s)

# children only text.
# child_texts = [
#     s["text"] for s in all_samples if s["speaker"] == "CHI"
# ]

# print(child_texts)

dialogue = []
for s in all_samples:
    dialogue.append(f"{s['speaker']}: {s['text']}")


# print(dialogue)

from datasets import Dataset, load_dataset

dataset = Dataset.from_dict({
    "text": dialogue
})

print(dataset)
print(dataset[0])

dataset.save_to_disk("child_dialogue_dataset")

from datasets import load_from_disk
dataset = load_from_disk("child_dialogue_dataset")

# # 1. Load Dataset
# dataset = load_dataset("suayptalha/Poetry-Foundation-Poems") 
# print(dataset)
# print(dataset["train"])

# dataset = dataset["train"].select(range(2000))
# splits = dataset.train_test_split(test_size=0.1)
