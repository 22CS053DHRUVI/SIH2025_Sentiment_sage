#!/usr/bin/env python3
"""
MCA eConsultation AI System with Model Saving/Loading
====================================================

This version saves models after training and loads them on subsequent runs
to avoid retraining every time.
"""

import os
import csv
from pathlib import Path
import pickle
import json

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


def load_saved_models():
    """Load saved models if they exist."""
    models_dir = Path("models")
    
    if not models_dir.exists():
        print("❌ No saved models found. Will train new models.")
        return None, None
    
    try:
        print("💾 Loading saved models...")
        
        # Load sentiment model
        with open(models_dir / "sentiment_model.pkl", 'rb') as f:
            sentiment_data = pickle.load(f)
        
        # Load summarizer model  
        with open(models_dir / "summarizer_model.pkl", 'rb') as f:
            summarizer_data = pickle.load(f)
        
        # Load keyword config
        with open(models_dir / "keyword_config.json", 'r') as f:
            keyword_config = json.load(f)
        
        print("✅ Saved models loaded successfully!")
        print(f"📅 Models saved at: {sentiment_data.get('saved_at', 'Unknown')}")
        
        return {
            'sentiment': sentiment_data,
            'summarizer': summarizer_data,
            'keywords': keyword_config
        }
        
    except Exception as e:
        print(f"❌ Error loading saved models: {e}")
        print("🔄 Will train new models instead.")
        return None


def create_models_from_saved(saved_models, cfg):
    """Create model instances from saved data."""
    print("🔧 Creating models from saved data...")
    
    # Create sentiment model
    sentiment = SentimentPipeline(
        cfg.sentiment.model_name,
        cfg.sentiment.max_length,
        cfg.device
    )
    
    # Load saved state
    sentiment.model.load_state_dict(saved_models['sentiment']['model_state'])
    sentiment.tokenizer = saved_models['sentiment']['tokenizer']
    
    # Create summarizer model
    summarizer = Summarizer(
        cfg.summarization.model_name,
        cfg.summarization.max_input_length,
        cfg.summarization.max_summary_length,
        cfg.summarization.num_beams,
        cfg.device
    )
    
    # Load saved state
    summarizer.model.load_state_dict(saved_models['summarizer']['model_state'])
    summarizer.tokenizer = saved_models['summarizer']['tokenizer']
    
    return sentiment, summarizer


def save_models_after_processing(sentiment, summarizer, cfg):
    """Save models after processing for future use."""
    print("💾 Saving models for future use...")
    
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save sentiment model
        sentiment_data = {
            'model_name': cfg.sentiment.model_name,
            'max_length': cfg.sentiment.max_length,
            'device': str(sentiment.device),
            'model_state': sentiment.model.state_dict(),
            'tokenizer': sentiment.tokenizer,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(models_dir / "sentiment_model.pkl", 'wb') as f:
            pickle.dump(sentiment_data, f)
        
        # Save summarizer model
        summarizer_data = {
            'model_name': cfg.summarization.model_name,
            'max_input_length': summarizer.max_input_len,
            'max_summary_length': summarizer.max_summary_len,
            'num_beams': summarizer.num_beams,
            'device': str(summarizer.device),
            'model_state': summarizer.model.state_dict(),
            'tokenizer': summarizer.tokenizer,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(models_dir / "summarizer_model.pkl", 'wb') as f:
            pickle.dump(summarizer_data, f)
        
        # Save keyword config
        keyword_config = {
            'method': cfg.keywords.method,
            'top_k': cfg.keywords.top_k,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(models_dir / "keyword_config.json", 'w') as f:
            json.dump(keyword_config, f, indent=2)
        
        print("✅ Models saved successfully!")
        
    except Exception as e:
        print(f"❌ Error saving models: {e}")


def main(config_path: str = "configs/default.yaml"):
    """Main function with model saving/loading."""
    from datetime import datetime
    
    print("🚀 Starting MCA AI Project with Model Saving/Loading...")
    print("=" * 60)
    
    cfg = load_config(config_path)
    exp_dir = os.path.join(cfg.paths.experiments_dir, "baseline")
    ensure_dir(exp_dir)

    # Load data
    print("📊 Loading dataset...")
    ds: DatasetDict = load_dataset_any(cfg)
    test_split = ds["test"]
    print(f"✅ Dataset loaded: {len(test_split)} test examples")

    texts = test_split["text"]
    
    # Try to load saved models first
    saved_models = load_saved_models()
    
    if saved_models:
        print("⚡ Using saved models for faster processing...")
        try:
            sentiment, summarizer = create_models_from_saved(saved_models, cfg)
            print("✅ Models loaded from saved data!")
        except Exception as e:
            print(f"❌ Error loading saved models: {e}")
            print("🔄 Training new models instead...")
            sentiment = SentimentPipeline(cfg.sentiment.model_name, cfg.sentiment.max_length, cfg.device)
            summarizer = Summarizer(cfg.summarization.model_name, cfg.summarization.max_input_length, cfg.summarization.max_summary_length, cfg.summarization.num_beams, cfg.device)
    else:
        print("🔄 Training new models...")
        sentiment = SentimentPipeline(cfg.sentiment.model_name, cfg.sentiment.max_length, cfg.device)
        summarizer = Summarizer(cfg.summarization.model_name, cfg.summarization.max_input_length, cfg.summarization.max_summary_length, cfg.summarization.num_beams, cfg.device)

    # Process data
    print("🔄 Processing data...")
    
    # Sentiment analysis
    print("📊 Running sentiment analysis...")
    pred_labels = sentiment.predict(texts, batch_size=cfg.sentiment.batch_size)
    print(f"✅ Sentiment analysis completed: {len(pred_labels)} predictions")
    
    # Summarization
    print("📝 Running summarization...")
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
    print("🔍 Extracting keywords...")
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
    print("💾 Saving results...")
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
    print("☁️ Generating word cloud...")
    try:
        wc = build_wordcloud(" ".join(texts), width=cfg.viz.wordcloud.width, height=cfg.viz.wordcloud.height, background_color=cfg.viz.wordcloud.background_color)
        wc_path = os.path.join(exp_dir, "wordcloud.png")
        wc.to_file(wc_path)
        print(f"✅ Word cloud saved: {wc_path}")
    except Exception as e:
        print(f"❌ Error generating word cloud: {e}")

    # Save models for future use
    save_models_after_processing(sentiment, summarizer, cfg)

    print("\n🎉 Project completed successfully!")
    print(f"📁 Results saved in: {exp_dir}")
    print("💡 Models saved for faster future runs!")


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


