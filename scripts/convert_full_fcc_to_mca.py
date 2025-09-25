#!/usr/bin/env python3
"""
Convert the entire FCC dataset (4,833 comments) to MCA eConsultation format.
This script transforms all FCC comments to look like MCA stakeholder responses.
"""

import pandas as pd
import random
import re
from pathlib import Path

def convert_full_fcc_to_mca():
    """Convert all FCC comments to MCA consultation format."""
    
    print("🔄 Converting entire FCC dataset to MCA eConsultation format...")
    print("📊 Processing 4,833 FCC comments...")
    
    # Load the full FCC dataset
    fcc_df = pd.read_csv("data/fcc/train.csv")
    print(f"✅ Loaded {len(fcc_df)} FCC comments")
    
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
        "Government Official",
        "Audit Firm",
        "Legal Firm",
        "Corporate Advisory",
        "Compliance Officer",
        "Financial Advisor"
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
        "Digital Compliance",
        "Insolvency and Bankruptcy Code",
        "SEBI Regulations",
        "Accounting Standards",
        "Tax Compliance",
        "Environmental Compliance",
        "Labor Law Compliance",
        "Data Protection",
        "Cybersecurity",
        "Risk Management",
        "Internal Controls"
    ]
    
    # MCA-specific transformation mappings
    mca_transformations = {
        # Core domain mappings
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
    
    # MCA-specific comment templates
    mca_comment_templates = [
        "I am writing to submit my comments on the proposed {topic} as a {stakeholder}. ",
        "As a {stakeholder}, I would like to express my views on the {topic}. ",
        "I submit these comments on behalf of {stakeholder} regarding the {topic}. ",
        "In my capacity as a {stakeholder}, I wish to comment on the {topic}. ",
        "I am a {stakeholder} and I have reviewed the proposed {topic}. ",
        "On behalf of {stakeholder}, I submit the following comments on {topic}. ",
        "I am writing as a {stakeholder} to provide feedback on the {topic}. ",
        "As a practicing {stakeholder}, I would like to comment on the {topic}. ",
        "I represent {stakeholder} and wish to submit comments on the {topic}. ",
        "In my professional capacity as a {stakeholder}, I comment on the {topic}. "
    ]
    
    # Convert each FCC comment
    mca_comments = []
    
    for idx, row in fcc_df.iterrows():
        if idx % 500 == 0:
            print(f"Processing comment {idx+1}/{len(fcc_df)}...")
        
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
        
        # Select stakeholder and topic
        stakeholder = random.choice(stakeholder_types)
        topic = random.choice(consultation_topics)
        
        # Create MCA-style comment
        template = random.choice(mca_comment_templates)
        mca_intro = template.format(stakeholder=stakeholder, topic=topic)
        
        # Clean up the transformed text
        transformed_text = transformed_text.replace("\\n", " ").replace("\\t", " ")
        transformed_text = re.sub(r'\s+', ' ', transformed_text).strip()
        
        # Create full MCA comment
        mca_comment = f"Subject: Comments on {topic}\n\n"
        mca_comment += f"Submitted by: {stakeholder}\n\n"
        mca_comment += f"Comment: {mca_intro}{transformed_text}\n\n"
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
            'character_count': len(mca_comment),
            'original_text': original_text[:200] + "..." if len(original_text) > 200 else original_text
        })
    
    # Create MCA dataset
    mca_df = pd.DataFrame(mca_comments)
    
    # Create output directory
    Path("data/mca").mkdir(parents=True, exist_ok=True)
    
    # Save MCA dataset
    mca_csv_path = "data/mca/mca_consultation_comments.csv"
    mca_df.to_csv(mca_csv_path, index=False)
    
    # Also save as train.csv for the project
    train_csv_path = "data/mca/train.csv"
    mca_df.to_csv(train_csv_path, index=False)
    
    print(f"\n✅ MCA eConsultation dataset created successfully!")
    print(f"📊 Total comments: {len(mca_df)}")
    print(f"📁 Files saved:")
    print(f"   - {mca_csv_path}")
    print(f"   - {train_csv_path}")
    
    # Show statistics
    print(f"\n📈 Dataset Statistics:")
    print(f"   - Stakeholder types: {mca_df['stakeholder_type'].nunique()}")
    print(f"   - Consultation topics: {mca_df['consultation_topic'].nunique()}")
    print(f"   - Average comment length: {mca_df['word_count'].mean():.0f} words")
    print(f"   - Average character count: {mca_df['character_count'].mean():.0f} characters")
    
    print(f"\n👥 Stakeholder Distribution:")
    stakeholder_counts = mca_df['stakeholder_type'].value_counts()
    for stakeholder, count in stakeholder_counts.head(10).items():
        print(f"   - {stakeholder}: {count} comments")
    
    print(f"\n📋 Consultation Topics Distribution:")
    topic_counts = mca_df['consultation_topic'].value_counts()
    for topic, count in topic_counts.head(10).items():
        print(f"   - {topic}: {count} comments")
    
    return mca_df

def main():
    print("🏛️  FCC to MCA eConsultation Dataset Converter")
    print("=" * 60)
    
    try:
        mca_df = convert_full_fcc_to_mca()
        print(f"\n🎉 Conversion completed successfully!")
        print(f"📊 Ready to use with MCA eConsultation AI system")
        print(f"🚀 Run: python mca_project.py")
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
