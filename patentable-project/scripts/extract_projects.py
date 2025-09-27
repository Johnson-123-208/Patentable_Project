#!/usr/bin/env python3
"""
Extract and clean student project abstracts from CSV or PDF files.
Output: data/clean/projects.csv
"""

import os
import re
import pandas as pd
import pdfplumber
import PyMuPDF
from pathlib import Path
import argparse
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class ProjectExtractor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """Clean and normalize text data."""
        if pd.isna(text) or text == '':
            return ''
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-]', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S*@\S*\s?', '', text)
        
        # Remove extra punctuation
        text = re.sub(r'[.]{2,}', '.', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_from_csv(self, file_path, id_col='id', title_col='title', abstract_col='abstract'):
        """Extract projects from CSV file."""
        print(f"Reading CSV file: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Handle different column name possibilities
        columns = df.columns.str.lower()
        
        # Map columns
        col_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'id' in col_lower:
                col_mapping['id'] = col
            elif 'title' in col_lower or 'name' in col_lower:
                col_mapping['title'] = col
            elif 'abstract' in col_lower or 'description' in col_lower or 'summary' in col_lower:
                col_mapping['abstract'] = col
        
        # Create standardized dataframe
        projects = []
        for idx, row in df.iterrows():
            project = {
                'id': row.get(col_mapping.get('id', df.columns[0]), f"proj_{idx+1}"),
                'title': self.clean_text(str(row.get(col_mapping.get('title', ''), ''))),
                'abstract': self.clean_text(str(row.get(col_mapping.get('abstract', ''), '')))
            }
            projects.append(project)
        
        return pd.DataFrame(projects)
    
    def extract_from_pdf(self, file_path):
        """Extract projects from PDF file."""
        print(f"Reading PDF file: {file_path}")
        
        projects = []
        
        try:
            # Try with pdfplumber first
            with pdfplumber.open(file_path) as pdf:
                full_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
        except:
            # Fallback to PyMuPDF
            doc = PyMuPDF.open(file_path)
            full_text = ""
            for page_num in range(doc.page_count):
                page = doc[page_num]
                full_text += page.get_text() + "\n"
            doc.close()
        
        # Simple parsing: assume projects are separated by patterns like "Project #" or similar
        project_sections = re.split(r'(?i)(?:project\s*#?\s*\d+|title:\s*|abstract:\s*)', full_text)
        
        for i, section in enumerate(project_sections[1:], 1):  # Skip first empty split
            if section.strip():
                lines = section.strip().split('\n')
                title = self.clean_text(lines[0][:100]) if lines else f"Project {i}"
                abstract = self.clean_text(' '.join(lines[1:]) if len(lines) > 1 else lines[0])
                
                projects.append({
                    'id': f"proj_{i}",
                    'title': title,
                    'abstract': abstract
                })
        
        return pd.DataFrame(projects)
    
    def process_file(self, input_path, output_path):
        """Process input file and save cleaned projects."""
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if input_path.suffix.lower() == '.csv':
            df = self.extract_from_csv(input_path)
        elif input_path.suffix.lower() == '.pdf':
            df = self.extract_from_pdf(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
        
        # Remove empty abstracts
        df = df[df['abstract'].str.len() > 10]
        
        # Add additional fields
        df['word_count'] = df['abstract'].apply(lambda x: len(x.split()) if x else 0)
        df['char_count'] = df['abstract'].apply(len)
        
        print(f"Extracted {len(df)} projects")
        print(f"Saving to: {output_path}")
        
        df.to_csv(output_path, index=False)
        return df


def main():
    parser = argparse.ArgumentParser(description='Extract and clean project data')
    parser.add_argument('--input', '-i', required=True, help='Input file path (CSV or PDF)')
    parser.add_argument('--output', '-o', default='data/clean/projects.csv', help='Output CSV path')
    
    args = parser.parse_args()
    
    extractor = ProjectExtractor()
    df = extractor.process_file(args.input, args.output)
    
    print(f"\nSample projects:")
    print(df[['id', 'title', 'word_count']].head())


if __name__ == "__main__":
    # If run without arguments, process sample data
    if len(os.sys.argv) == 1:
        print("Processing sample data...")
        extractor = ProjectExtractor()
        df = extractor.process_file('data/sample/projects.csv', 'data/clean/projects.csv')
        print(f"\nProcessed {len(df)} projects from sample data")
    else:
        main()