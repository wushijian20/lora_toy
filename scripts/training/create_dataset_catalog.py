"""
Build a comprehensive catalog of 1000 datasets for LoRA training.
This script creates datasets_catalog.csv with metadata for all 1000 datasets.
"""

import pandas as pd
import os
from datetime import datetime

def create_1000_dataset_catalog():
    """Create catalog for 1000 datasets"""
    
    # Existing 30 datasets you're using
    base_datasets = {
        'babylm-100M-children-stories': ('deven367/babylm-100M-children-stories', 'literature', 'narrative'),
        'bedtime_stories': ('gofilipa/bedtime_stories', 'literature', 'narrative'),
        'Shakespeare_Corpus': ('sarnab/Shakespeare_Corpus', 'literature', 'poetic'),
        'KingJamesVersionBible': ('SzuTao/KingJamesVersionBible', 'literature', 'biblical'),
        'old_english_dataset': ('azizsi/old_english_dataset', 'literature', 'archaic'),
        'victorian_authorship': ('contemmcm/victorian_authorship', 'literature', 'victorian'),
        'poetry-chinese-zhtw': ('erhwenkuo/poetry-chinese-zhtw', 'literature', 'poetic'),
        'fairy_tales': ('aslicu/fairy_tales', 'literature', 'narrative'),
        'Mythological': ('AJ69/Mythological', 'literature', 'mythological'),
        'ChatPILE-Casual': ('Smilyai-labs/ChatPILE-Casual', 'social_media', 'casual'),
        'corporate-speak-dataset': ('phxdev/corporate-speak-dataset', 'business', 'formal'),
        'research-paper-abstracts': ('sumukshashidhar-testing/research-paper-abstracts', 'academic', 'technical'),
        'neurips-2024-peer-reviews': ('Samarth0710/neurips-2024-peer-reviews', 'academic', 'academic'),
        'reddit_finance_posts_sp500': ('emilpartow/reddit_finance_posts_sp500', 'finance', 'technical'),
        'reddit-sarcasm': ('Thewillonline/reddit-sarcasm', 'social_media', 'sarcastic'),
        'reddit-logic': ('agentlans/reddit-logic', 'social_media', 'logical'),
        'reddit_top_comments': ('cowWhySo/reddit_top_comments', 'social_media', 'social'),
        'reddit-blogspot-twitter': ('jonaskoenig/reddit-blogspot-twitter', 'social_media', 'mixed'),
        'reddit_autism_dataset': ('Osondu/reddit_autism_dataset', 'social_media', 'technical'),
        'covid_dialogue': ('Tlighteval/covid_dialogue', 'dialogue', 'dialogue'),
        'tv_dialogue': ('sedthh/tv_dialogue', 'dialogue', 'dialogue'),
        'American_English_Natural_Dialogue_Speech_Data': ('Nexdata/American_English_Natural_Dialogue_Speech_Data', 'dialogue', 'dialogue'),
        'medical_dialogue-chinese-zhtw': ('erhwenkuo/medical_dialogue-chinese-zhtw', 'medical', 'medical'),
        'dialy_dialogue_with_recoginized_concept_raw': ('jpeandrew/dialy_dialogue_with_recoginized_concept_raw', 'dialogue', 'dialogue'),
        'soccer-dialogues': ('rony/soccer-dialogues', 'dialogue', 'dialogue'),
        'empathetic_dialogues_for_lm': ('pixelsandpointers/empathetic_dialogues_for_lm', 'dialogue', 'empathetic'),
        'empathetic_dialogues_v2': ('Adapting/empathetic_dialogues_v2', 'dialogue', 'empathetic'),
        'Poetry-Foundation-Poems': ('suayptalha/Poetry-Foundation-Poems', 'literature', 'poetic'),
    }
    
    # Additional datasets to reach 1000
    # These should be real HuggingFace datasets or custom sources
    additional_datasets = {
        'wikitext': ('wikitext', 'encyclopedic', 'formal'),
        'wikitext-2': ('wikitext', 'encyclopedic', 'formal'),
        'wikitext-103': ('wikitext', 'encyclopedic', 'formal'),
        'openwebtext': ('openwebtext', 'web', 'mixed'),
        'cc_news': ('cc_news', 'news', 'journalistic'),
        'gigaword': ('gigaword', 'news', 'summarized'),
        'xsum': ('xsum', 'news', 'summarized'),
        'news_commentary': ('news_commentary', 'news', 'journalistic'),
        'europarl': ('europarl', 'political', 'formal'),
        'wikibio': ('wikibio', 'biographical', 'narrative'),
    }
    
    # Combine datasets
    all_datasets = {**base_datasets}
    
    # Generate catalog DataFrame
    catalog_data = []
    dataset_id = 1
    
    for name, (hf_path, domain, style) in all_datasets.items():
        catalog_data.append({
            'dataset_id': dataset_id,
            'name': name,
            'hf_path': hf_path,
            'domain': domain,
            'style': style,
            'status': 'pending',
            'token_count': 0,
            'training_time_minutes': 0,
            'perplexity': 0.0,
            'created_at': datetime.now().isoformat()
        })
        dataset_id += 1
    
    # Generate additional placeholder datasets to reach 1000
    for i in range(len(all_datasets), 1000):
        domain_options = ['literature', 'social_media', 'academic', 'news', 'dialogue', 'technical', 'business']
        style_options = ['formal', 'casual', 'poetic', 'narrative', 'technical', 'dialogue', 'sarcastic']
        
        domain = domain_options[i % len(domain_options)]
        style = style_options[i % len(style_options)]
        
        catalog_data.append({
            'dataset_id': dataset_id,
            'name': f'dataset_{dataset_id:04d}_{domain}_{style}',
            'hf_path': f'custom/dataset_{dataset_id:04d}',  # Placeholder
            'domain': domain,
            'style': style,
            'status': 'pending',
            'token_count': 0,
            'training_time_minutes': 0,
            'perplexity': 0.0,
            'created_at': datetime.now().isoformat()
        })
        dataset_id += 1
    
    # Create DataFrame
    df = pd.DataFrame(catalog_data)
    
    # Save catalog
    catalog_path = 'datasets_catalog.csv'
    df.to_csv(catalog_path, index=False)
    
    print(f"✅ Created catalog with {len(df)} datasets")
    print(f"📊 Domain distribution:")
    print(df['domain'].value_counts())
    print(f"🎨 Style distribution:")
    print(df['style'].value_counts())
    print(f"📁 Saved to: {catalog_path}")
    
    return df

def validate_catalog(catalog_path='datasets_catalog.csv'):
    """Validate catalog integrity"""
    df = pd.read_csv(catalog_path)
    
    print("\n📋 Catalog Validation:")
    print(f"Total datasets: {len(df)}")
    print(f"Unique domains: {df['domain'].nunique()}")
    print(f"Unique styles: {df['style'].nunique()}")
    print(f"Pending: {(df['status'] == 'pending').sum()}")
    print(f"Completed: {(df['status'] == 'completed').sum()}")
    print(f"Failed: {(df['status'] == 'failed').sum()}")
    
    # Check for duplicates
    if df['name'].duplicated().any():
        print("⚠️ Warning: Duplicate dataset names found")
    
    return df

if __name__ == '__main__':
    create_1000_dataset_catalog()
    validate_catalog()
