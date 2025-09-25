#!/usr/bin/env python3
"""
Transform FCC comments to simulate MCA eConsultation responses.
This script converts telecom policy comments to corporate law consultation format.
"""

import pandas as pd
import random
import re
from pathlib import Path

def transform_fcc_to_mca(input_csv: str, output_csv: str):
    """Transform FCC comments to MCA consultation format."""
    
    print("🔄 Transforming FCC comments to MCA eConsultation format...")
    
    # Load FCC data
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} FCC comments")
    
    # MCA-specific transformation mappings
    mca_transformations = {
        # Policy domain mappings
        "net neutrality": "corporate governance",
        "internet freedom": "regulatory compliance",
        "ISP regulations": "MCA regulations",
        "broadband access": "corporate access",
        "telecommunications": "corporate law",
        "FCC": "MCA",
        "federal communications commission": "ministry of corporate affairs",
        "internet service providers": "corporate entities",
        "consumers": "stakeholders",
        "public interest": "corporate interest",
        "competition": "market competition",
        "innovation": "business innovation",
        "investment": "corporate investment",
        "infrastructure": "corporate infrastructure",
        "privacy": "data privacy",
        "security": "corporate security",
        "transparency": "regulatory transparency",
        "accountability": "corporate accountability",
        "oversight": "regulatory oversight",
        "enforcement": "compliance enforcement",
        "violations": "compliance violations",
        "penalties": "regulatory penalties",
        "licensing": "corporate licensing",
        "permits": "regulatory permits",
        "standards": "compliance standards",
        "guidelines": "regulatory guidelines",
        "procedures": "compliance procedures",
        "requirements": "regulatory requirements",
        "obligations": "compliance obligations",
        "duties": "corporate duties",
        "responsibilities": "corporate responsibilities",
        "rights": "stakeholder rights",
        "protections": "regulatory protections",
        "safeguards": "compliance safeguards",
        "monitoring": "regulatory monitoring",
        "reporting": "compliance reporting",
        "disclosure": "corporate disclosure",
        "transparency": "regulatory transparency",
        "accountability": "corporate accountability",
        "governance": "corporate governance",
        "management": "corporate management",
        "leadership": "corporate leadership",
        "board": "board of directors",
        "directors": "board members",
        "shareholders": "stakeholders",
        "investors": "corporate investors",
        "creditors": "corporate creditors",
        "employees": "corporate employees",
        "customers": "corporate customers",
        "suppliers": "corporate suppliers",
        "partners": "business partners",
        "vendors": "corporate vendors",
        "contractors": "corporate contractors",
        "subsidiaries": "corporate subsidiaries",
        "affiliates": "corporate affiliates",
        "associates": "corporate associates",
        "joint ventures": "corporate joint ventures",
        "mergers": "corporate mergers",
        "acquisitions": "corporate acquisitions",
        "restructuring": "corporate restructuring",
        "reorganization": "corporate reorganization",
        "liquidation": "corporate liquidation",
        "insolvency": "corporate insolvency",
        "bankruptcy": "corporate bankruptcy",
        "winding up": "corporate winding up",
        "dissolution": "corporate dissolution",
        "incorporation": "corporate incorporation",
        "registration": "corporate registration",
        "filing": "corporate filing",
        "submission": "regulatory submission",
        "application": "regulatory application",
        "approval": "regulatory approval",
        "consent": "regulatory consent",
        "permission": "regulatory permission",
        "authorization": "regulatory authorization",
        "license": "corporate license",
        "permit": "regulatory permit",
        "certificate": "regulatory certificate",
        "registration": "corporate registration",
        "incorporation": "corporate incorporation",
        "amendment": "regulatory amendment",
        "modification": "regulatory modification",
        "revision": "regulatory revision",
        "update": "regulatory update",
        "change": "regulatory change",
        "reform": "regulatory reform",
        "modernization": "regulatory modernization",
        "simplification": "regulatory simplification",
        "streamlining": "regulatory streamlining",
        "efficiency": "regulatory efficiency",
        "effectiveness": "regulatory effectiveness",
        "compliance": "regulatory compliance",
        "adherence": "regulatory adherence",
        "conformity": "regulatory conformity",
        "observance": "regulatory observance",
        "implementation": "regulatory implementation",
        "execution": "regulatory execution",
        "enforcement": "regulatory enforcement",
        "monitoring": "regulatory monitoring",
        "supervision": "regulatory supervision",
        "oversight": "regulatory oversight",
        "inspection": "regulatory inspection",
        "audit": "regulatory audit",
        "review": "regulatory review",
        "assessment": "regulatory assessment",
        "evaluation": "regulatory evaluation",
        "analysis": "regulatory analysis",
        "examination": "regulatory examination",
        "investigation": "regulatory investigation",
        "inquiry": "regulatory inquiry",
        "probe": "regulatory probe",
        "scrutiny": "regulatory scrutiny",
        "surveillance": "regulatory surveillance",
        "tracking": "regulatory tracking",
        "monitoring": "regulatory monitoring",
        "reporting": "regulatory reporting",
        "disclosure": "corporate disclosure",
        "transparency": "regulatory transparency",
        "accountability": "corporate accountability",
        "responsibility": "corporate responsibility",
        "liability": "corporate liability",
        "obligation": "corporate obligation",
        "duty": "corporate duty",
        "responsibility": "corporate responsibility",
        "accountability": "corporate accountability",
        "liability": "corporate liability",
        "obligation": "corporate obligation",
        "duty": "corporate duty",
        "responsibility": "corporate responsibility",
        "accountability": "corporate accountability",
        "liability": "corporate liability",
        "obligation": "corporate obligation",
        "duty": "corporate duty"
    }
    
    # MCA stakeholder types
    stakeholder_types = [
        "Chartered Accountant",
        "Company Secretary",
        "Corporate Lawyer",
        "Industry Association",
        "Corporate Entity",
        "Public Citizen",
        "Regulatory Professional",
        "Business Consultant",
        "Academic Expert",
        "Government Official"
    ]
    
    # MCA consultation topics
    consultation_topics = [
        "Companies Act Amendment",
        "Corporate Governance Guidelines",
        "Audit Requirements",
        "Board Composition Rules",
        "Disclosure Norms",
        "Compliance Procedures",
        "Regulatory Framework",
        "Stakeholder Protection",
        "Corporate Social Responsibility",
        "Digital Compliance"
    ]
    
    # Transform each comment
    mca_comments = []
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"Processing comment {idx+1}/{len(df)}")
        
        # Get original text
        original_text = row['text']
        
        # Apply transformations
        transformed_text = original_text
        
        # Replace FCC-specific terms with MCA terms
        for fcc_term, mca_term in mca_transformations.items():
            transformed_text = re.sub(
                re.escape(fcc_term), 
                mca_term, 
                transformed_text, 
                flags=re.IGNORECASE
            )
        
        # Add MCA-specific context
        topic = random.choice(consultation_topics)
        stakeholder = random.choice(stakeholder_types)
        
        # Create MCA-style comment
        mca_comment = f"Subject: Comments on {topic}\n\n"
        mca_comment += f"Submitted by: {stakeholder}\n\n"
        mca_comment += f"Comment: {transformed_text}\n\n"
        mca_comment += f"I urge the MCA to consider these observations while finalizing the {topic.lower()}."
        
        # Create MCA comment record
        mca_comments.append({
            'id': f"mca_{idx+1:06d}",
            'text': mca_comment,
            'source': 'mca_consultation',
            'stakeholder_type': stakeholder,
            'consultation_topic': topic,
            'original_fcc_id': row['id'],
            'word_count': len(mca_comment.split()),
            'character_count': len(mca_comment)
        })
    
    # Create MCA dataset
    mca_df = pd.DataFrame(mca_comments)
    
    # Save MCA dataset
    mca_df.to_csv(output_csv, index=False)
    
    print(f"✅ Created MCA consultation dataset: {output_csv}")
    print(f"✅ Comments: {len(mca_df)}")
    print(f"✅ Stakeholder types: {mca_df['stakeholder_type'].nunique()}")
    print(f"✅ Consultation topics: {mca_df['consultation_topic'].nunique()}")
    print(f"✅ Average comment length: {mca_df['word_count'].mean():.0f} words")
    
    return mca_df

def main():
    input_csv = "data/fcc/train.csv"
    output_csv = "data/mca/mca_consultation_comments.csv"
    
    # Create output directory
    Path("data/mca").mkdir(parents=True, exist_ok=True)
    
    # Transform data
    mca_df = transform_fcc_to_mca(input_csv, output_csv)
    
    print(f"\n🎯 MCA eConsultation dataset ready!")
    print(f"📁 File: {output_csv}")
    print(f"📊 Sample stakeholder types:")
    print(mca_df['stakeholder_type'].value_counts().head())
    print(f"\n📊 Sample consultation topics:")
    print(mca_df['consultation_topic'].value_counts().head())

if __name__ == "__main__":
    main()
