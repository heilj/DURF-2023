from datasets import load_dataset, load_from_disk

dataset = load_from_disk("top_quality_dataset_4000")

print((dataset['train']['x']))