import os
import csv
from pathlib import Path

import pandas as pd
from datasets import DatasetDict

from mca_ai.config import load_config
from mca_ai.data_loader import load_dataset_any
from mca_ai.models.sentiment import SentimentPipeline
from mca_ai.models.summarizer import Summarizer
from mca_ai.models.keywords import extract_keywords
from mca_ai.viz.wordcloud_utils import build_wordcloud


def ensure_dir(p: str):
	Path(p).mkdir(parents=True, exist_ok=True)


def main(config_path: str = "configs/default.yaml"):
	cfg = load_config(config_path)
	exp_dir = os.path.join(cfg.paths.experiments_dir, "baseline")
	ensure_dir(exp_dir)

	# Load data
	print("Loading dataset...")
	ds: DatasetDict = load_dataset_any(cfg)
	test_split = ds["test"]
	print(f"Dataset loaded: {len(test_split)} test examples")

	texts = test_split["text"]
	
	# Process in smaller batches to avoid memory issues
	batch_size = min(100, len(texts))  # Process in smaller batches
	
	print("Initializing sentiment model...")
	sent = SentimentPipeline(cfg.sentiment.model_name, cfg.sentiment.max_length, cfg.device)
	print("Running sentiment analysis...")
	pred_labels = sent.predict(texts, batch_size=cfg.sentiment.batch_size)
	print(f"Sentiment analysis completed: {len(pred_labels)} predictions")
	
	print("Initializing summarization model...")
	sumz = Summarizer(cfg.summarization.model_name, cfg.summarization.max_input_length, cfg.summarization.max_summary_length, cfg.summarization.num_beams, cfg.device)
	print("Running summarization...")
	summaries = []
	for i, text in enumerate(texts):
		if i % 50 == 0:
			print(f"Summarizing progress: {i}/{len(texts)}")
		try:
			summary = sumz.summarize(text)
			summaries.append(summary)
		except Exception as e:
			print(f"Error summarizing text {i}: {e}")
			summaries.append("Error in summarization")
	print(f"Summarization completed: {len(summaries)} summaries")
	
	print("Extracting keywords...")
	keywords_list = []
	for i, text in enumerate(texts):
		if i % 50 == 0:
			print(f"Keywords progress: {i}/{len(texts)}")
		try:
			keywords = extract_keywords(text, top_k=cfg.keywords.top_k)
			keywords_list.append("; ".join(keywords))
		except Exception as e:
			print(f"Error extracting keywords for text {i}: {e}")
			keywords_list.append("Error in keyword extraction")
	print(f"Keyword extraction completed: {len(keywords_list)} extractions")

	# Save CSV
	print("Saving results to CSV...")
	rows = []
	for i, t in enumerate(texts):
		rows.append({
			"text": t,
			"sentiment": pred_labels[i],
			"summary": summaries[i],
			"keywords": keywords_list[i],
		})
	df = pd.DataFrame(rows)
	csv_path = os.path.join(exp_dir, "predictions.csv")
	df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
	print(f"✓ Saved predictions: {csv_path}")

	# Word cloud from full corpus
	print("Generating word cloud...")
	try:
		wc = build_wordcloud(" ".join(texts), width=cfg.viz.wordcloud.width, height=cfg.viz.wordcloud.height, background_color=cfg.viz.wordcloud.background_color)
		wc_path = os.path.join(exp_dir, "wordcloud.png")
		wc.to_file(wc_path)
		print(f"✓ Saved word cloud: {wc_path}")
	except Exception as e:
		print(f"Error generating word cloud: {e}")

	print("\n🎉 Project completed successfully!")
	print(f"Results saved in: {exp_dir}")


if __name__ == "__main__":
	try:
		print("🚀 Starting MCA AI Project...")
		main()
	except KeyboardInterrupt:
		print("\n⚠️  Project interrupted by user")
	except Exception as e:
		print(f"\n❌ Project failed with error: {e}")
		import traceback
		traceback.print_exc()
		print("\n💡 Try running with a smaller dataset or check your configuration")
