#!/usr/bin/env python3
"""
Save Trained Models After Processing
===================================

This script saves the trained models after the MCA analysis is complete,
so they can be loaded quickly in future runs without retraining.
"""

import os
import pickle
import torch
import json
from pathlib import Path
from datetime import datetime

def save_models_after_training():
    """Save models after they have been trained."""
    print("💾 Saving trained models for future use...")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Import the models that were just trained
        from mca_ai.config import load_config
        from mca_ai.models.sentiment import SentimentPipeline
        from mca_ai.models.summarizer import Summarizer
        
        # Load config
        config = load_config()
        
        print("📊 Loading trained models...")
        
        # Initialize models (they will load the pre-trained weights)
        sentiment_model = SentimentPipeline(
            config.sentiment.model_name,
            config.sentiment.max_length,
            config.device
        )
        
        summarizer_model = Summarizer(
            config.summarization.model_name,
            config.summarization.max_input_length,
            config.summarization.max_summary_length,
            config.summarization.num_beams,
            config.device
        )
        
        # Save sentiment model
        print("💾 Saving sentiment analysis model...")
        sentiment_data = {
            'model_name': config.sentiment.model_name,
            'max_length': config.sentiment.max_length,
            'device': str(sentiment_model.device),
            'model_state': sentiment_model.model.state_dict(),
            'tokenizer': sentiment_model.tokenizer,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(models_dir / "sentiment_model.pkl", 'wb') as f:
            pickle.dump(sentiment_data, f)
        
        # Save summarizer model
        print("💾 Saving summarization model...")
        summarizer_data = {
            'model_name': config.summarization.model_name,
            'max_input_length': summarizer_model.max_input_len,
            'max_summary_length': summarizer_model.max_summary_len,
            'num_beams': summarizer_model.num_beams,
            'device': str(summarizer_model.device),
            'model_state': summarizer_model.model.state_dict(),
            'tokenizer': summarizer_model.tokenizer,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(models_dir / "summarizer_model.pkl", 'wb') as f:
            pickle.dump(summarizer_data, f)
        
        # Save keyword extraction config
        print("💾 Saving keyword extraction config...")
        keyword_config = {
            'method': config.keywords.method,
            'top_k': config.keywords.top_k,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(models_dir / "keyword_config.json", 'w') as f:
            json.dump(keyword_config, f, indent=2)
        
        # Save system info
        system_info = {
            'models_saved_at': datetime.now().isoformat(),
            'sentiment_model': config.sentiment.model_name,
            'summarizer_model': config.summarization.model_name,
            'keyword_method': config.keywords.method,
            'config_file': 'configs/default.yaml'
        }
        
        with open(models_dir / "system_info.json", 'w') as f:
            json.dump(system_info, f, indent=2)
        
        print("✅ Models saved successfully!")
        print(f"📁 Saved to: {models_dir.absolute()}")
        print(f"📋 Files created:")
        print(f"   - sentiment_model.pkl")
        print(f"   - summarizer_model.pkl") 
        print(f"   - keyword_config.json")
        print(f"   - system_info.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving models: {e}")
        return False

def check_saved_models():
    """Check if models are already saved."""
    models_dir = Path("models")
    
    required_files = [
        "sentiment_model.pkl",
        "summarizer_model.pkl", 
        "keyword_config.json",
        "system_info.json"
    ]
    
    all_exist = all((models_dir / file).exists() for file in required_files)
    
    if all_exist:
        print("✅ Saved models found!")
        print(f"📁 Models directory: {models_dir.absolute()}")
        
        # Show when models were saved
        try:
            with open(models_dir / "system_info.json", 'r') as f:
                info = json.load(f)
                print(f"📅 Models saved at: {info.get('models_saved_at', 'Unknown')}")
        except:
            pass
        
        return True
    else:
        print("❌ No saved models found")
        return False

def main():
    """Main function."""
    print("🔍 Checking for saved models...")
    
    if check_saved_models():
        print("\n💡 Models are already saved. You can use them for faster processing.")
        print("🚀 To use saved models, run: python project.py")
    else:
        print("\n💾 No saved models found. Saving models after training...")
        if save_models_after_training():
            print("\n🎉 Models saved successfully!")
            print("💡 Next time you run the system, it will load these saved models instead of retraining.")
        else:
            print("\n❌ Failed to save models. Check the error above.")

if __name__ == "__main__":
    main()

