#!/usr/bin/env python3
"""
Create a sample FCC dataset for testing the MCA AI project.
This creates a realistic sample based on FCC comment structure.
"""

import os
import pandas as pd
import random
from pathlib import Path

def create_fcc_sample(output_dir: str = "data/fcc", num_samples: int = 1000):
    """Create a realistic sample of FCC comments."""
    print(f"Creating FCC comment sample with {num_samples} examples...")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Sample FCC comment templates based on real FCC comment patterns
    positive_comments = [
        "I strongly support this proposed rule. It will help ensure fair competition in the telecommunications market.",
        "This is an excellent initiative that will benefit consumers and promote innovation.",
        "I fully endorse this proposal. It addresses the key concerns raised by stakeholders.",
        "This rule will provide much-needed protection for consumers and small businesses.",
        "I applaud the Commission for taking this important step forward.",
        "This proposal strikes the right balance between innovation and consumer protection.",
        "I support this measure as it will promote broadband deployment in rural areas.",
        "This is a well-thought-out approach that will benefit the entire industry.",
        "I strongly agree with the proposed changes to the regulatory framework.",
        "This initiative will help bridge the digital divide and ensure universal access."
    ]
    
    negative_comments = [
        "I oppose this proposed rule as it will stifle innovation and competition.",
        "This regulation is unnecessary and will create burdensome compliance costs.",
        "I strongly disagree with this approach. It will harm consumers in the long run.",
        "This proposal goes too far and will discourage investment in new technologies.",
        "I cannot support this rule as it will create market distortions.",
        "This regulation is overly complex and will be difficult to implement effectively.",
        "I oppose this measure as it will reduce consumer choice and competition.",
        "This proposal will create unintended consequences that harm the industry.",
        "I disagree with this approach as it will slow down technological progress.",
        "This rule will create unnecessary barriers to entry for new market participants."
    ]
    
    neutral_comments = [
        "I have mixed feelings about this proposed rule and need more information.",
        "This is a complex issue that requires careful consideration of all factors.",
        "I need to see more details about the implementation before forming an opinion.",
        "This proposal raises important questions that deserve thorough analysis.",
        "I have concerns about certain aspects but see merit in other parts.",
        "This rule needs further refinement to address stakeholder concerns.",
        "I appreciate the effort but have questions about the practical implementation.",
        "This is a nuanced issue that requires balanced consideration.",
        "I need more time to review the full implications of this proposal.",
        "This regulation touches on many complex issues that need careful study."
    ]
    
    # Generate sample data
    data = []
    for i in range(num_samples):
        # Randomly select sentiment and corresponding comment
        sentiment = random.choice(['positive', 'negative', 'neutral'])
        if sentiment == 'positive':
            comment = random.choice(positive_comments)
        elif sentiment == 'negative':
            comment = random.choice(negative_comments)
        else:
            comment = random.choice(neutral_comments)
        
        # Add some variation and context
        comment += f" This is comment #{i+1} regarding the proposed telecommunications regulations."
        
        # Add some additional context based on comment type
        if i % 10 == 0:
            comment += " I am a telecommunications industry professional with 15 years of experience."
        elif i % 15 == 0:
            comment += " I represent a consumer advocacy group and speak on behalf of our members."
        elif i % 20 == 0:
            comment += " I am a small business owner who relies on telecommunications services."
        
        data.append({
            'id': i + 1,
            'text': comment,
            'label': sentiment,
            'source': 'fcc_sample',
            'date': f"2023-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            'author_type': random.choice(['individual', 'organization', 'business'])
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "fcc_comments.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"✓ Created FCC sample dataset: {csv_path}")
    print(f"✓ Rows: {len(df)}")
    print(f"✓ Columns: {list(df.columns)}")
    print(f"✓ Sentiment distribution:")
    print(df['label'].value_counts())
    
    return csv_path

def main():
    csv_path = create_fcc_sample()
    print(f"\nTo use this FCC sample data, update configs/default.yaml:")
    print(f"  data.source: csv")
    print(f"  data.text_field: text")
    print(f"  paths.data_dir: ./data/fcc")
    print(f"  data.fast_limit: 500  # or remove this line to use all data")

if __name__ == "__main__":
    main()
