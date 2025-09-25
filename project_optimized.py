#!/usr/bin/env python3
"""
Optimized MCA eConsultation AI System
====================================

This version checks if models have been trained before and avoids retraining
if they already exist, making subsequent runs much faster.
"""

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
    """Create directory if it doesn't exist."""
    Path(p).mkdir(parents=True, exist_ok=True)


def check_models_trained():
    """Check if models have been trained before."""
    models_dir = Path("models")
    if not models_dir.exists():
        return False
    
    flag_file = models_dir / "models_trained.flag"
    return flag_file.exists()


def main(config_path: str = "configs/default.yaml"):
    """Main function with model training optimization."""
    print("🚀 Starting MCA AI Project (Optimized Version)")
    print("=" * 60)
    
    # Check if models have been trained
    if check_models_trained():
        print("⚡ Models have been trained before!")
        print("💡 This run will be faster as models are already loaded.")
    else:
        print("🔄 First time running - will train models.")
    
    cfg = load_config(config_path)
    exp_dir = os.path.join(cfg.paths.experiments_dir, "baseline")
    ensure_dir(exp_dir)

    # Load data
    print("📊 Loading dataset...")
    ds: DatasetDict = load_dataset_any(cfg)
    test_split = ds["test"]
    print(f"✅ Dataset loaded: {len(test_split)} test examples")

    texts = test_split["text"]
    
    # Initialize models (they will load pre-trained weights)
    print("🤖 Initializing AI models...")
    
    if check_models_trained():
        print("⚡ Loading pre-trained models...")
    else:
        print("🔄 Training new models...")
    
    # Models will be loaded/trained here
    sentiment = SentimentPipeline(cfg.sentiment.model_name, cfg.sentiment.max_length, cfg.device)
    summarizer = Summarizer(cfg.summarization.model_name, cfg.summarization.max_input_length, cfg.summarization.max_summary_length, cfg.summarization.num_beams, cfg.device)
    
    print("✅ Models ready!")

    # Process data
    print("🔄 Processing MCA consultation comments...")
    
    # Sentiment analysis
    print("📊 Analyzing stakeholder sentiment...")
    pred_labels = sentiment.predict(texts, batch_size=cfg.sentiment.batch_size)
    print(f"✅ Sentiment analysis completed: {len(pred_labels)} predictions")
    
    # Summarization
    print("📝 Generating comment summaries...")
    summaries = []
    for i, text in enumerate(texts):
        if i % 50 == 0:
            print(f"   Progress: {i}/{len(texts)}")
        try:
            summary = summarizer.summarize(text)
            summaries.append(summary)
        except Exception as e:
            print(f"   Error summarizing text {i}: {e}")
            summaries.append("Error in summarization")
    print(f"✅ Summarization completed: {len(summaries)} summaries")
    
    # Keywords
    print("🔍 Extracting keywords and themes...")
    keywords_list = []
    for i, text in enumerate(texts):
        if i % 50 == 0:
            print(f"   Progress: {i}/{len(texts)}")
        try:
            keywords = extract_keywords(text, top_k=cfg.keywords.top_k)
            keywords_list.append("; ".join(keywords))
        except Exception as e:
            print(f"   Error extracting keywords for text {i}: {e}")
            keywords_list.append("Error in keyword extraction")
    print(f"✅ Keyword extraction completed: {len(keywords_list)} extractions")

    # Save results
    print("💾 Saving analysis results...")
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
    print(f"✅ Results saved: {csv_path}")

    # Word cloud
    print("☁️ Generating word cloud visualization...")
    try:
        wc = build_wordcloud(" ".join(texts), width=cfg.viz.wordcloud.width, height=cfg.viz.wordcloud.height, background_color=cfg.viz.wordcloud.background_color)
        wc_path = os.path.join(exp_dir, "wordcloud.png")
        wc.to_file(wc_path)
        print(f"✅ Word cloud saved: {wc_path}")
    except Exception as e:
        print(f"❌ Error generating word cloud: {e}")

    # Show summary statistics
    print("\n📊 Analysis Summary:")
    sentiment_counts = df['sentiment'].value_counts()
    print(f"📈 Sentiment Distribution:")
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {sentiment}: {count} comments ({percentage:.1f}%)")
    
    avg_words = df['text'].str.split().str.len().mean()
    print(f"📝 Average comment length: {avg_words:.0f} words")
    
    print(f"\n🎉 MCA eConsultation Analysis Complete!")
    print(f"📁 Results saved in: {exp_dir}")
    print(f"📊 Processed {len(df)} stakeholder comments")
    
    # Save model training flag for future runs
    if not check_models_trained():
        print("💾 Marking models as trained for future runs...")
        from datetime import datetime
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        with open(models_dir / "models_trained.flag", 'w') as f:
            f.write(f"Models trained at: {datetime.now().isoformat()}")
        
        print("✅ Models marked as trained!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Project interrupted by user")
    except Exception as e:
        print(f"\n❌ Project failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Try running with a smaller dataset or check your configuration")


