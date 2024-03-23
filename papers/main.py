from datasets import load_dataset

dataset = load_dataset("imagenet-1k")

print(dataset['train']['image'][0])

