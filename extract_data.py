from datasets import load_dataset

dataset = load_dataset("deven367/babylm-100M-children-stories")
print(dataset)
print(dataset['train'][0:10])
