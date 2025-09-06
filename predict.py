#!/usr/bin/env python3
"""
Semantic Column Classifier

Usage:
    python predict.py --input data.csv --column column_name

This script classifies a column into one of 5 semantic types:
1. Phone Number
2. Company Name  
3. Country
4. Date
5. Other

Classification is based purely on semantic similarity using embeddings.
"""

import argparse
import sys
import os
from utils import SemanticClassifier, load_column_from_file, sample_column_values


def main():
    parser = argparse.ArgumentParser(description='Classify column semantically using embeddings')
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    parser.add_argument('--column', required=True, help='Name of column to classify')
    parser.add_argument('--training-data', default='TrainingData/TrainingData', 
                       help='Path to training data directory')
    
    args = parser.parse_args()
    
    try:
        # Load the column data
        print(f"Loading column '{args.column}' from '{args.input}'...")
        column_values = load_column_from_file(args.input, args.column)
        
        # Sample values for classification
        sampled_values = sample_column_values(column_values, sample_size=100)
        
        print(f"Loaded {len(column_values)} values, using {len(sampled_values)} for classification")
        
        # Initialize classifier
        print("Initializing semantic classifier...")
        classifier = SemanticClassifier(training_data_path=args.training_data)
        
        # Classify the column
        print("Classifying column...")
        predicted_type = classifier.classify_column(sampled_values)
        
        # Map internal names to display names
        type_mapping = {
            'phone': 'Phone Number',
            'company': 'Company Name',
            'country': 'Country', 
            'date': 'Date',
            'other': 'Other'
        }
        
        display_type = type_mapping.get(predicted_type, 'Other')
        
        print(f"\nColumn Type: {display_type}")
        
        # Show some sample values for verification
        print(f"\nSample values:")
        for i, value in enumerate(sampled_values[:5]):
            print(f"  {i+1}. {value}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
