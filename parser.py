#!/usr/bin/env python3
"""
Semantic Column Parser

Usage:
    python parser.py --input data.csv

This script:
1. Automatically detects Phone Number and Company Name columns using semantic classification
2. Parses detected columns using semantic similarity:
   - Phone Numbers: [OriginalValue, Country, Number] 
   - Company Names: [OriginalValue, Name, Legal]
3. Outputs results to output.csv

All parsing is done using embeddings and cosine similarity - no regex or rules.
"""

import argparse
import sys
import os
import pandas as pd
from utils import SemanticClassifier, sample_column_values


def detect_columns_to_parse(df: pd.DataFrame, classifier: SemanticClassifier) -> dict:
    """Detect which columns are Phone Numbers or Company Names"""
    columns_to_parse = {
        'phone': [],
        'company': []
    }
    
    print("Analyzing columns for semantic types...")
    
    for column in df.columns:
        print(f"  Analyzing column: {column}")
        
        # Get sample values for classification
        column_values = df[column].tolist()
        sampled_values = sample_column_values(column_values, sample_size=50)
        
        # Classify the column
        predicted_type = classifier.classify_column(sampled_values)
        
        print(f"    Predicted type: {predicted_type}")
        
        if predicted_type == 'phone':
            columns_to_parse['phone'].append(column)
        elif predicted_type == 'company':
            columns_to_parse['company'].append(column)
    
    return columns_to_parse


def parse_phone_columns(df: pd.DataFrame, phone_columns: list, classifier: SemanticClassifier) -> pd.DataFrame:
    """Parse phone number columns into Country and Number"""
    results = []
    
    for column in phone_columns:
        print(f"\nParsing phone column: {column}")
        
        for idx, value in enumerate(df[column]):
            if pd.isna(value) or str(value).strip() == '':
                # Handle empty values
                results.append({
                    'OriginalValue': str(value),
                    'Country': '',
                    'Number': '',
                    'SourceColumn': column,
                    'RowIndex': idx
                })
            else:
                try:
                    country, number = classifier.parse_phone_number(str(value))
                    results.append({
                        'OriginalValue': str(value),
                        'Country': country,
                        'Number': number,
                        'SourceColumn': column,
                        'RowIndex': idx
                    })
                except Exception as e:
                    print(f"    Warning: Could not parse '{value}': {e}")
                    results.append({
                        'OriginalValue': str(value),
                        'Country': '',
                        'Number': str(value),
                        'SourceColumn': column,
                        'RowIndex': idx
                    })
    
    return pd.DataFrame(results)


def parse_company_columns(df: pd.DataFrame, company_columns: list, classifier: SemanticClassifier) -> pd.DataFrame:
    """Parse company name columns into Name and Legal"""
    results = []
    
    for column in company_columns:
        print(f"\nParsing company column: {column}")
        
        for idx, value in enumerate(df[column]):
            if pd.isna(value) or str(value).strip() == '':
                # Handle empty values
                results.append({
                    'OriginalValue': str(value),
                    'Name': '',
                    'Legal': '',
                    'SourceColumn': column,
                    'RowIndex': idx
                })
            else:
                try:
                    name, legal = classifier.parse_company_name(str(value))
                    results.append({
                        'OriginalValue': str(value),
                        'Name': name,
                        'Legal': legal,
                        'SourceColumn': column,
                        'RowIndex': idx
                    })
                except Exception as e:
                    print(f"    Warning: Could not parse '{value}': {e}")
                    results.append({
                        'OriginalValue': str(value),
                        'Name': str(value),
                        'Legal': '',
                        'SourceColumn': column,
                        'RowIndex': idx
                    })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Parse columns semantically using embeddings')
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    parser.add_argument('--output', default='output.csv', help='Path to output CSV file')
    parser.add_argument('--training-data', default='TrainingData/TrainingData', 
                       help='Path to training data directory')
    
    args = parser.parse_args()
    
    try:
        # Load the input file
        print(f"Loading data from '{args.input}'...")
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
        
        # Initialize classifier
        print("\nInitializing semantic classifier...")
        classifier = SemanticClassifier(training_data_path=args.training_data)
        
        # Detect columns that need parsing
        columns_to_parse = detect_columns_to_parse(df, classifier)
        
        phone_columns = columns_to_parse['phone']
        company_columns = columns_to_parse['company']
        
        print(f"\nFound columns to parse:")
        print(f"  Phone columns: {phone_columns}")
        print(f"  Company columns: {company_columns}")
        
        if not phone_columns and not company_columns:
            print("\nNo Phone Number or Company Name columns detected. Nothing to parse.")
            return
        
        # Parse the detected columns
        all_results = []
        
        if phone_columns:
            phone_results = parse_phone_columns(df, phone_columns, classifier)
            phone_results['Type'] = 'Phone'
            all_results.append(phone_results)
        
        if company_columns:
            company_results = parse_company_columns(df, company_columns, classifier)
            company_results['Type'] = 'Company'
            all_results.append(company_results)
        
        # Combine results
        if all_results:
            final_results = pd.concat(all_results, ignore_index=True)
            
            # Save to output file
            final_results.to_csv(args.output, index=False)
            print(f"\nResults saved to '{args.output}'")
            print(f"Parsed {len(final_results)} total values")
            
            # Show summary
            print(f"\nSummary:")
            for result_type in final_results['Type'].unique():
                type_count = len(final_results[final_results['Type'] == result_type])
                print(f"  {result_type}: {type_count} values")
            
            # Show sample results
            print(f"\nSample results:")
            print(final_results.head(10).to_string(index=False))
        else:
            print("\nNo results to save.")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
