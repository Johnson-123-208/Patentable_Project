#!/usr/bin/env python3
"""
Extract and normalize patent data from CSV files.
Output: data/clean/patents.csv
"""

import os
import re
import pandas as pd
from pathlib import Path
import argparse
import numpy as np


class PatentExtractor:
    def __init__(self):
        pass
        
    def clean_text(self, text):
        """Clean and normalize patent text data."""
        if pd.isna(text) or text == '':
            return ''
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Clean up patent-specific noise
        text = re.sub(r'\bfig\.\s*\d+\b', '', text)  # Remove figure references
        text = re.sub(r'\bclaim\s*\d+\b', '', text)  # Remove claim references
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def normalize_ipc_codes(self, ipc_string):
        """Normalize IPC classification codes."""
        if pd.isna(ipc_string) or ipc_string == '':
            return ''
        
        # Split by common delimiters
        codes = re.split(r'[;,\|]', str(ipc_string))
        
        # Clean each code
        clean_codes = []
        for code in codes:
            code = code.strip().upper()
            # Basic IPC format validation (e.g., A01B1/00)
            if re.match(r'^[A-H]\d{2}[A-Z]\d+/\d+', code):
                clean_codes.append(code)
        
        return '; '.join(clean_codes)
    
    def extract_main_ipc_class(self, ipc_codes):
        """Extract main IPC class (first letter and two digits)."""
        if not ipc_codes:
            return ''
        
        codes = ipc_codes.split(';')
        if codes:
            main_code = codes[0].strip()
            if len(main_code) >= 3:
                return main_code[:3]  # e.g., A01
        return ''
    
    def process_csv(self, input_path, output_path):
        """Process patent CSV file and save cleaned data."""
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Reading patent CSV: {input_path}")
        df = pd.read_csv(input_path)
        
        # Standardize column names
        columns = df.columns.str.lower()
        col_mapping = {}
        
        for col in df.columns:
            col_lower = col.lower()
            if 'id' in col_lower or 'patent' in col_lower and 'id' in col_lower:
                col_mapping['id'] = col
            elif 'title' in col_lower:
                col_mapping['title'] = col
            elif 'abstract' in col_lower:
                col_mapping['abstract'] = col
            elif 'claim' in col_lower:
                col_mapping['claims'] = col
            elif 'ipc' in col_lower or 'classification' in col_lower:
                col_mapping['ipc_codes'] = col
            elif 'link' in col_lower or 'url' in col_lower:
                col_mapping['link'] = col
        
        # Process patents
        patents = []
        for idx, row in df.iterrows():
            patent = {
                'id': str(row.get(col_mapping.get('id', df.columns[0]), f"pat_{idx+1}")),
                'title': self.clean_text(str(row.get(col_mapping.get('title', ''), ''))),
                'abstract': self.clean_text(str(row.get(col_mapping.get('abstract', ''), ''))),
                'claims': self.clean_text(str(row.get(col_mapping.get('claims', ''), ''))),
                'ipc_codes': self.normalize_ipc_codes(row.get(col_mapping.get('ipc_codes', ''), '')),
                'link': str(row.get(col_mapping.get('link', ''), ''))
            }
            
            # Add derived features
            patent['main_ipc_class'] = self.extract_main_ipc_class(patent['ipc_codes'])
            patent['abstract_length'] = len(patent['abstract'])
            patent['claims_length'] = len(patent['claims'])
            patent['word_count'] = len(patent['abstract'].split()) if patent['abstract'] else 0
            
            # Combine text for embedding
            combined_text = f"{patent['title']} {patent['abstract']} {patent['claims']}"
            patent['combined_text'] = self.clean_text(combined_text)
            
            patents.append(patent)
        
        # Create DataFrame
        clean_df = pd.DataFrame(patents)
        
        # Remove patents with very short abstracts
        clean_df = clean_df[clean_df['abstract_length'] >= 50]
        
        print(f"Processed {len(clean_df)} patents")
        print(f"Saving to: {output_path}")
        
        clean_df.to_csv(output_path, index=False)
        
        # Print summary statistics
        print(f"\nPatent Statistics:")
        print(f"Average abstract length: {clean_df['abstract_length'].mean():.0f} chars")
        print(f"Average word count: {clean_df['word_count'].mean():.0f} words")
        print(f"Unique IPC classes: {clean_df['main_ipc_class'].nunique()}")
        
        return clean_df


def main():
    parser = argparse.ArgumentParser(description='Extract and normalize patent data')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file path')
    parser.add_argument('--output', '-o', default='data/clean/patents.csv', help='Output CSV path')
    
    args = parser.parse_args()
    
    extractor = PatentExtractor()
    df = extractor.process_csv(args.input, args.output)
    
    print(f"\nSample patents:")
    print(df[['id', 'title', 'main_ipc_class', 'word_count']].head())


if __name__ == "__main__":
    # If run without arguments, process sample data
    if len(os.sys.argv) == 1:
        print("Processing sample data...")
        extractor = PatentExtractor()
        df = extractor.process_csv('data/sample/patents.csv', 'data/clean/patents.csv')
        print(f"\nProcessed {len(df)} patents from sample data")
    else:
        main()