#!/usr/bin/env python3
"""
Column Classification and Scoring Tool

This script analyzes each column in your dataset and provides:
1. Classification type for each column
2. Prediction confidence scores
3. Sample analysis
4. Detailed metrics

Usage:
    python test_columns.py --input "VIT TESTCASES - Sheet1.csv"
"""

import argparse
import pandas as pd
import numpy as np
from utils import SemanticClassifier, sample_column_values
import json
import os
from datetime import datetime


class ColumnAnalyzer:
    """Analyze and classify all columns in a dataset"""
    
    def __init__(self, training_data_path: str = "TrainingData/TrainingData"):
        print("Initializing semantic classifier...")
        self.classifier = SemanticClassifier(training_data_path)
        self.category_mapping = {
            'phone': 'Phone Number',
            'company': 'Company Name', 
            'country': 'Country',
            'date': 'Date',
            'other': 'Other'
        }
    
    def analyze_single_column(self, column_data, column_name):
        """Analyze a single column and return detailed results"""
        print(f"\nAnalyzing column: {column_name}")
        
        # Clean and filter data
        valid_values = [str(val) for val in column_data if pd.notna(val) and str(val).strip()]
        total_values = len(column_data)
        valid_count = len(valid_values)
        empty_count = total_values - valid_count
        
        if not valid_values:
            return {
                'column_name': column_name,
                'predicted_type': 'Other',
                'confidence_scores': {},
                'total_values': total_values,
                'valid_values': 0,
                'empty_values': empty_count,
                'sample_values': [],
                'classification_details': 'No valid values found'
            }
        
        # Sample values for classification
        sampled_values = sample_column_values(valid_values, sample_size=min(100, len(valid_values)))
        
        # Get prediction
        predicted_category = self.classifier.classify_column(sampled_values)
        predicted_type = self.category_mapping.get(predicted_category, 'Other')
        
        # Calculate confidence scores for all categories
        if sampled_values:
            # Get mean embedding of sampled values
            value_embeddings = self.classifier.embed_text(sampled_values)
            mean_embedding = np.mean(value_embeddings, axis=0)
            
            # Calculate similarities with all prototypes
            confidence_scores = {}
            for category, prototype in self.classifier.category_prototypes.items():
                similarity = self.classifier.compute_similarity(mean_embedding, prototype)
                display_name = self.category_mapping.get(category, category)
                confidence_scores[display_name] = float(similarity)
        else:
            confidence_scores = {}
        
        # Get diverse sample values
        sample_size = min(10, len(valid_values))
        if len(valid_values) > sample_size:
            # Try to get diverse samples
            step = len(valid_values) // sample_size
            sample_indices = [i * step for i in range(sample_size)]
            sample_values = [valid_values[i] for i in sample_indices]
        else:
            sample_values = valid_values[:sample_size]
        
        return {
            'column_name': column_name,
            'predicted_type': predicted_type,
            'confidence_scores': confidence_scores,
            'total_values': total_values,
            'valid_values': valid_count,
            'empty_values': empty_count,
            'sample_values': sample_values,
            'classification_details': f'Classified as {predicted_type} with confidence {confidence_scores.get(predicted_type, 0):.3f}'
        }
    
    def analyze_dataset(self, file_path):
        """Analyze all columns in the dataset"""
        print(f"Loading dataset: {file_path}")
        df = pd.read_csv(file_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns to analyze: {len(df.columns)}")
        
        results = []
        
        for column_name in df.columns:
            column_result = self.analyze_single_column(df[column_name], column_name)
            results.append(column_result)
            
            # Print immediate results
            print(f"  Result: {column_result['predicted_type']} "
                  f"(confidence: {column_result['confidence_scores'].get(column_result['predicted_type'], 0):.3f})")
        
        return results, df
    
    def generate_summary_report(self, results, output_dir="column_analysis_results"):
        """Generate comprehensive analysis report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create detailed JSON report
        json_file = os.path.join(output_dir, "column_analysis.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'analysis_timestamp': datetime.now().isoformat(),
                'total_columns': len(results),
                'results': results
            }, f, indent=2, ensure_ascii=False)
        
        # Create summary CSV
        summary_data = []
        for result in results:
            row = {
                'Column_Name': result['column_name'],
                'Predicted_Type': result['predicted_type'],
                'Confidence_Score': result['confidence_scores'].get(result['predicted_type'], 0),
                'Total_Values': result['total_values'],
                'Valid_Values': result['valid_values'],
                'Empty_Values': result['empty_values'],
                'Sample_Value_1': result['sample_values'][0] if result['sample_values'] else '',
                'Sample_Value_2': result['sample_values'][1] if len(result['sample_values']) > 1 else '',
                'Sample_Value_3': result['sample_values'][2] if len(result['sample_values']) > 2 else ''
            }
            
            # Add confidence scores for all categories
            for category in self.category_mapping.values():
                row[f'Confidence_{category.replace(" ", "_")}'] = result['confidence_scores'].get(category, 0)
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        csv_file = os.path.join(output_dir, "column_classification_summary.csv")
        summary_df.to_csv(csv_file, index=False)
        
        # Create detailed text report
        report_file = os.path.join(output_dir, "detailed_analysis_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("SEMANTIC COLUMN CLASSIFICATION ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Columns Analyzed: {len(results)}\n\n")
            
            # Type distribution
            type_counts = {}
            for result in results:
                pred_type = result['predicted_type']
                type_counts[pred_type] = type_counts.get(pred_type, 0) + 1
            
            f.write("CLASSIFICATION DISTRIBUTION:\n")
            f.write("-" * 30 + "\n")
            for type_name, count in sorted(type_counts.items()):
                percentage = (count / len(results)) * 100
                f.write(f"{type_name}: {count} columns ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Detailed column analysis
            f.write("DETAILED COLUMN ANALYSIS:\n")
            f.write("-" * 30 + "\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"{i}. Column: {result['column_name']}\n")
                f.write(f"   Predicted Type: {result['predicted_type']}\n")
                f.write(f"   Primary Confidence: {result['confidence_scores'].get(result['predicted_type'], 0):.3f}\n")
                f.write(f"   Data Quality: {result['valid_values']}/{result['total_values']} valid values\n")
                
                f.write("   Confidence Scores:\n")
                for category, score in sorted(result['confidence_scores'].items(), key=lambda x: x[1], reverse=True):
                    f.write(f"     {category}: {score:.3f}\n")
                
                f.write("   Sample Values:\n")
                for j, sample in enumerate(result['sample_values'][:5], 1):
                    f.write(f"     {j}. {sample}\n")
                f.write("\n")
        
        # Create high-confidence predictions file
        high_confidence_results = [r for r in results if r['confidence_scores'].get(r['predicted_type'], 0) > 0.5]
        if high_confidence_results:
            hc_file = os.path.join(output_dir, "high_confidence_predictions.csv")
            hc_data = []
            for result in high_confidence_results:
                hc_data.append({
                    'Column_Name': result['column_name'],
                    'Predicted_Type': result['predicted_type'],
                    'Confidence_Score': result['confidence_scores'].get(result['predicted_type'], 0),
                    'Sample_Values': ' | '.join(result['sample_values'][:3])
                })
            pd.DataFrame(hc_data).to_csv(hc_file, index=False)
        
        print(f"\nAnalysis complete!")
        print(f"Results saved to: {output_dir}/")
        print(f"Files created:")
        print(f"  - {csv_file}")
        print(f"  - {json_file}")
        print(f"  - {report_file}")
        if high_confidence_results:
            print(f"  - {hc_file}")
        
        return summary_df
    
    def print_summary_statistics(self, results):
        """Print quick summary statistics"""
        print("\n" + "="*60)
        print("CLASSIFICATION SUMMARY")
        print("="*60)
        
        # Type distribution
        type_counts = {}
        confidence_by_type = {}
        
        for result in results:
            pred_type = result['predicted_type']
            confidence = result['confidence_scores'].get(pred_type, 0)
            
            type_counts[pred_type] = type_counts.get(pred_type, 0) + 1
            if pred_type not in confidence_by_type:
                confidence_by_type[pred_type] = []
            confidence_by_type[pred_type].append(confidence)
        
        for type_name, count in sorted(type_counts.items()):
            avg_confidence = np.mean(confidence_by_type[type_name])
            percentage = (count / len(results)) * 100
            print(f"{type_name}: {count} columns ({percentage:.1f}%) - Avg Confidence: {avg_confidence:.3f}")
        
        # High confidence predictions
        high_conf_count = sum(1 for r in results if r['confidence_scores'].get(r['predicted_type'], 0) > 0.5)
        print(f"\nHigh Confidence Predictions (>0.5): {high_conf_count}/{len(results)} ({(high_conf_count/len(results)*100):.1f}%)")
        
        # Data quality summary
        total_values = sum(r['total_values'] for r in results)
        valid_values = sum(r['valid_values'] for r in results)
        print(f"Overall Data Quality: {valid_values}/{total_values} ({(valid_values/total_values*100):.1f}%) valid values")


def main():
    parser = argparse.ArgumentParser(description='Analyze and classify all columns in a dataset')
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    parser.add_argument('--output-dir', default='column_analysis_results',
                       help='Directory to save analysis results')
    parser.add_argument('--training-data', default='TrainingData/TrainingData',
                       help='Path to training data directory')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = ColumnAnalyzer(training_data_path=args.training_data)
        
        # Analyze dataset
        results, df = analyzer.analyze_dataset(args.input)
        
        # Generate comprehensive report
        summary_df = analyzer.generate_summary_report(results, args.output_dir)
        
        # Print summary statistics
        analyzer.print_summary_statistics(results)
        
        print(f"\nFor detailed results, check: {args.output_dir}/")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
