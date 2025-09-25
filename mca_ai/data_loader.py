import os
import shutil
import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset
from datasets.exceptions import NonMatchingSplitsSizesError

from mca_ai.preprocess import clean_text


def clear_dataset_cache(dataset_name: str):
	"""Clear the cache for a specific dataset to resolve metadata mismatches."""
	try:
		from datasets import get_dataset_infos
		# Get cache directory
		cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
		dataset_cache_dir = os.path.join(cache_dir, dataset_name.replace("/", "___"))
		
		if os.path.exists(dataset_cache_dir):
			shutil.rmtree(dataset_cache_dir)
			print(f"Cleared cache for dataset: {dataset_name}")
	except Exception as e:
		print(f"Warning: Could not clear cache for {dataset_name}: {e}")


def load_dataset_with_fallback(app_cfg) -> DatasetDict:
	"""Load dataset with multiple fallback strategies for Hugging Face datasets."""
	dataset_name = app_cfg.data.hf_dataset
	
	# Strategy 1: Normal load
	try:
		print(f"Attempting to load {dataset_name} normally...")
		return load_dataset(dataset_name)
	except NonMatchingSplitsSizesError as e:
		print(f"Split size mismatch: {e}")
	except Exception as e:
		print(f"Normal load failed: {e}")
	
	# Strategy 2: Clear cache and force redownload
	try:
		print("Clearing cache and forcing redownload...")
		clear_dataset_cache(dataset_name)
		return load_dataset(dataset_name, download_mode="force_redownload")
	except Exception as e:
		print(f"Force redownload failed: {e}")
	
	# Strategy 3: Ignore verifications
	try:
		print("Loading with ignore_verifications=True...")
		return load_dataset(dataset_name, ignore_verifications=True)
	except Exception as e:
		print(f"Ignore verifications failed: {e}")
	
	# Strategy 4: Load specific splits individually
	try:
		print("Attempting to load splits individually...")
		# Try to load train split first
		train_ds = load_dataset(dataset_name, split="train", ignore_verifications=True)
		# Try to load test split if it exists
		try:
			test_ds = load_dataset(dataset_name, split="test", ignore_verifications=True)
		except:
			# If no test split, create one from train
			train_test_split = train_ds.train_test_split(test_size=app_cfg.data.split_ratio[1], seed=app_cfg.seed)
			train_ds = train_test_split["train"]
			test_ds = train_test_split["test"]
		
		return DatasetDict({"train": train_ds, "test": test_ds})
	except Exception as e:
		print(f"Individual split loading failed: {e}")
	
	raise RuntimeError(f"All strategies failed to load dataset: {dataset_name}")


def create_synthetic_dataset(num_examples: int = 1000) -> DatasetDict:
	"""Create a synthetic dataset for testing when real data is not available."""
	import random
	
	# Sample texts for different sentiment categories
	positive_texts = [
		"I love this new policy! It's exactly what we need for our community.",
		"This is a fantastic initiative that will benefit everyone.",
		"Great work on this proposal. I fully support it.",
		"Excellent idea! This will make a real difference.",
		"I'm very happy with this decision. Thank you for listening.",
		"This is wonderful news for our city.",
		"I strongly support this measure. It's well thought out.",
		"Perfect! This addresses all my concerns.",
		"Outstanding work! This is exactly what we needed.",
		"I'm thrilled about this development."
	]
	
	negative_texts = [
		"This policy is terrible and will hurt our community.",
		"I strongly oppose this initiative. It's a bad idea.",
		"This proposal is poorly thought out and will cause problems.",
		"I'm very disappointed with this decision.",
		"This is a waste of taxpayer money.",
		"I cannot support this measure. It's harmful.",
		"This is a terrible idea that will backfire.",
		"I'm outraged by this proposal.",
		"This will make things worse, not better.",
		"I strongly disagree with this approach."
	]
	
	neutral_texts = [
		"I have mixed feelings about this policy.",
		"This seems like a reasonable approach, but I have some concerns.",
		"I need more information before I can form an opinion.",
		"This is an interesting proposal that deserves consideration.",
		"I see both pros and cons to this initiative.",
		"This is a complex issue that requires careful analysis.",
		"I'm not sure if this is the right solution.",
		"This could work, but there might be better alternatives.",
		"I have questions about the implementation details.",
		"This is worth discussing further."
	]
	
	# Generate synthetic data
	texts = []
	labels = []
	
	for i in range(num_examples):
		sentiment = random.choice(['positive', 'negative', 'neutral'])
		if sentiment == 'positive':
			text = random.choice(positive_texts)
		elif sentiment == 'negative':
			text = random.choice(negative_texts)
		else:
			text = random.choice(neutral_texts)
		
		# Add some variation
		text += f" This is comment #{i+1} about the proposed policy."
		texts.append(text)
		labels.append(sentiment)
	
	# Create dataset
	dataset = Dataset.from_dict({
		"text": texts,
		"label": labels,
		"id": list(range(num_examples))
	})
	
	# Split into train/test
	train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
	
	return DatasetDict({
		"train": train_test_split["train"],
		"test": train_test_split["test"]
	})


def load_dataset_any(app_cfg) -> DatasetDict:
	"""Load dataset based on config source in app_cfg.

	Returns a DatasetDict with 'train' and 'test' splits. Adds a cleaned 'text' field.
	"""
	source = app_cfg.data.source
	ds = None
	
	if source == "hf_remote":
		try:
			ds = load_dataset_with_fallback(app_cfg)
		except Exception as e:
			print(f"Hugging Face dataset loading failed: {e}")
			print("Falling back to synthetic dataset for testing...")
			ds = create_synthetic_dataset(1000)
	elif source == "local_json":
		json_path = f"{app_cfg.paths.data_dir}/train.jsonl"
		if os.path.exists(json_path):
			ds = load_dataset("json", data_files={"train": json_path})
		else:
			print(f"Local JSON file not found: {json_path}")
			print("Falling back to synthetic dataset...")
			ds = create_synthetic_dataset(1000)
	elif source == "local_parquet":
		parquet_path = f"{app_cfg.paths.data_dir}/train.parquet"
		if os.path.exists(parquet_path):
			ds = load_dataset("parquet", data_files={"train": parquet_path})
		else:
			print(f"Local Parquet file not found: {parquet_path}")
			print("Falling back to synthetic dataset...")
			ds = create_synthetic_dataset(1000)
	elif source == "csv":
		csv_path = f"{app_cfg.paths.data_dir}/train.csv"
		if os.path.exists(csv_path):
			ds = load_dataset("csv", data_files={"train": csv_path})
		else:
			print(f"Local CSV file not found: {csv_path}")
			print("Falling back to synthetic dataset...")
			ds = create_synthetic_dataset(1000)
	elif source == "synthetic":
		print("Using synthetic dataset for testing...")
		ds = create_synthetic_dataset(1000)
	else:
		raise ValueError(f"Unknown data source: {source}")

	if ds is None:
		raise RuntimeError("Failed to load dataset")

	# Ensure splits
	if isinstance(ds, DatasetDict):
		if "train" in ds and "test" not in ds:
			parts = ds["train"].train_test_split(test_size=app_cfg.data.split_ratio[1], seed=app_cfg.seed)
			ds = DatasetDict({"train": parts["train"], "test": parts["test"]})
	else:
		ds = DatasetDict({"train": ds["train"], "test": ds.get("test", ds["train"].train_test_split(test_size=app_cfg.data.split_ratio[1], seed=app_cfg.seed)["test"])})

	# Apply fast limit if specified
	if hasattr(app_cfg.data, 'fast_limit') and app_cfg.data.fast_limit is not None:
		for split_name in ds.keys():
			if len(ds[split_name]) > app_cfg.data.fast_limit:
				print(f"Limiting {split_name} split to {app_cfg.data.fast_limit} examples")
				ds[split_name] = ds[split_name].select(range(app_cfg.data.fast_limit))

	# Clean and standardize text field
	orig_field = app_cfg.data.text_field
	def _map_clean(example):
		text = example.get(orig_field, "")
		return {"text": clean_text(text)}

	# Apply text cleaning with parallel processing
	map_num_proc = getattr(app_cfg.data, 'map_num_proc', 4)
	ds = ds.map(_map_clean, num_proc=map_num_proc)
	
	return ds
