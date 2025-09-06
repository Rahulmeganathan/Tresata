#!/usr/bin/env python3
"""
Model Evaluation and Metrics Generator

This script evaluates the semantic classification model performance and generates
accuracy metrics, confusion matrix, and detailed performance reports.

Usage:
    python evaluate_model.py --test-data test_data.csv
    python evaluate_model.py --generate-test-set
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from utils import SemanticClassifier, sample_column_values
import json
import os


class ModelEvaluator:
    """Evaluate the semantic classification model performance"""
    
    def __init__(self, training_data_path: str = "TrainingData/TrainingData"):
        self.classifier = SemanticClassifier(training_data_path)
        self.category_mapping = {
            'phone': 'Phone Number',
            'company': 'Company Name', 
            'country': 'Country',
            'date': 'Date',
            'other': 'Other'
        }
    
    def generate_test_dataset(self, output_file: str = "evaluation_test_data.csv"):
        """Generate a labeled test dataset from training data"""
        print("Generating labeled test dataset...")
        
        test_data = []
        
        # Sample from each category
        categories = {
            'Phone Number': self.classifier.phone_data[:50],
            'Company Name': self.classifier.company_data[:50],
            'Country': self.classifier.countries_data[:50],
            'Date': self.classifier.dates_data[:50]
        }
        
        # Add some "Other" data (mixed/random data)
        other_data = [
            "random text 123", "email@example.com", "SKU-12345", 
            "product description", "user comment", "id_value_789",
            "measurement 45.6", "category_name", "status_active",
            "code ABC123", "reference REF001", "note: important"
        ]
        categories['Other'] = other_data
        
        for true_label, data_samples in categories.items():
            for sample in data_samples:
                if sample and str(sample).strip():
                    test_data.append({
                        'value': str(sample).strip(),
                        'true_label': true_label,
                        'category': true_label.lower().replace(' ', '_').replace('_number', '').replace('_name', '')
                    })
        
        # Create DataFrame and save
        test_df = pd.DataFrame(test_data)
        test_df = test_df.sample(frac=1).reset_index(drop=True)  # Shuffle
        test_df.to_csv(output_file, index=False)
        
        print(f"Generated test dataset with {len(test_df)} samples")
        print(f"Label distribution:")
        print(test_df['true_label'].value_counts())
        print(f"Saved to: {output_file}")
        
        return test_df
    
    def evaluate_model(self, test_data_file: str):
        """Evaluate model performance on test dataset"""
        print(f"Loading test data from: {test_data_file}")
        
        if not os.path.exists(test_data_file):
            print(f"Test file not found. Generating new test dataset...")
            test_df = self.generate_test_dataset(test_data_file)
        else:
            test_df = pd.read_csv(test_data_file)
        
        print(f"Evaluating {len(test_df)} samples...")
        
        # Make predictions
        predictions = []
        true_labels = []
        prediction_scores = []
        
        for idx, row in test_df.iterrows():
            value = row['value']
            true_label = row['true_label']
            
            # Classify single value
            predicted_category = self.classifier.classify_column([value], sample_size=1)
            predicted_label = self.category_mapping.get(predicted_category, 'Other')
            
            # Get confidence scores
            value_embedding = self.classifier.embed_text([value])[0]
            similarities = {}
            for category, prototype in self.classifier.category_prototypes.items():
                similarity = self.classifier.compute_similarity(value_embedding, prototype)
                similarities[self.category_mapping.get(category, category)] = similarity
            
            predictions.append(predicted_label)
            true_labels.append(true_label)
            prediction_scores.append(similarities)
            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(test_df)} samples")
        
        return true_labels, predictions, prediction_scores, test_df
    
    def calculate_metrics(self, true_labels, predictions):
        """Calculate various performance metrics"""
        # Overall accuracy
        accuracy = accuracy_score(true_labels, predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average=None, labels=list(self.category_mapping.values())
        )
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions, labels=list(self.category_mapping.values()))
        
        metrics = {
            'overall_accuracy': accuracy,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'per_class_metrics': {},
            'confusion_matrix': cm.tolist(),
            'class_labels': list(self.category_mapping.values())
        }
        
        # Per-class metrics
        for i, label in enumerate(self.category_mapping.values()):
            if i < len(precision):
                metrics['per_class_metrics'][label] = {
                    'precision': float(precision[i]) if not np.isnan(precision[i]) else 0.0,
                    'recall': float(recall[i]) if not np.isnan(recall[i]) else 0.0,
                    'f1_score': float(f1[i]) if not np.isnan(f1[i]) else 0.0,
                    'support': int(support[i])
                }
        
        return metrics
    
    def generate_report(self, metrics, prediction_scores, output_dir: str = "evaluation_results"):
        """Generate comprehensive evaluation report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics as JSON
        metrics_file = os.path.join(output_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Generate text report
        report_file = os.path.join(output_dir, "evaluation_report.txt")
        with open(report_file, 'w') as f:
            f.write("SEMANTIC CLASSIFICATION MODEL EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Overall Accuracy: {metrics['overall_accuracy']:.3f}\n")
            f.write(f"Weighted Precision: {metrics['precision_weighted']:.3f}\n")
            f.write(f"Weighted Recall: {metrics['recall_weighted']:.3f}\n")
            f.write(f"Weighted F1-Score: {metrics['f1_weighted']:.3f}\n\n")
            
            f.write("Per-Class Performance:\n")
            f.write("-" * 30 + "\n")
            for class_name, class_metrics in metrics['per_class_metrics'].items():
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {class_metrics['precision']:.3f}\n")
                f.write(f"  Recall: {class_metrics['recall']:.3f}\n")
                f.write(f"  F1-Score: {class_metrics['f1_score']:.3f}\n")
                f.write(f"  Support: {class_metrics['support']}\n\n")
        
        # Generate confusion matrix plot
        self.plot_confusion_matrix(metrics, output_dir)
        
        # Generate metrics summary plot
        self.plot_metrics_summary(metrics, output_dir)
        
        print(f"\nEvaluation complete!")
        print(f"Results saved to: {output_dir}/")
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.1%}")
        
        return metrics_file, report_file
    
    def plot_confusion_matrix(self, metrics, output_dir):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        cm = np.array(metrics['confusion_matrix'])
        labels = metrics['class_labels']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix - Semantic Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        cm_file = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved: {cm_file}")
    
    def plot_metrics_summary(self, metrics, output_dir):
        """Plot metrics summary"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Per-class metrics bar plot
        classes = list(metrics['per_class_metrics'].keys())
        precision_scores = [metrics['per_class_metrics'][c]['precision'] for c in classes]
        recall_scores = [metrics['per_class_metrics'][c]['recall'] for c in classes]
        f1_scores = [metrics['per_class_metrics'][c]['f1_score'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        ax1.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
        ax1.bar(x, recall_scores, width, label='Recall', alpha=0.8)
        ax1.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Score')
        ax1.set_title('Per-Class Performance Metrics')
        ax1.set_xticks(x)
        ax1.set_xticklabels(classes, rotation=45)
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # Overall metrics pie chart
        overall_scores = [
            metrics['overall_accuracy'],
            metrics['precision_weighted'], 
            metrics['recall_weighted'],
            metrics['f1_weighted']
        ]
        labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        ax2.pie(overall_scores, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Overall Weighted Metrics')
        
        plt.tight_layout()
        
        metrics_file = os.path.join(output_dir, "metrics_summary.png")
        plt.savefig(metrics_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Metrics summary saved: {metrics_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate semantic classification model')
    parser.add_argument('--test-data', default='evaluation_test_data.csv',
                       help='Path to test dataset CSV file')
    parser.add_argument('--generate-test-set', action='store_true',
                       help='Generate a new test dataset')
    parser.add_argument('--training-data', default='TrainingData/TrainingData',
                       help='Path to training data directory')
    parser.add_argument('--output-dir', default='evaluation_results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(training_data_path=args.training_data)
    
    if args.generate_test_set:
        # Just generate test set and exit
        evaluator.generate_test_dataset(args.test_data)
        return
    
    try:
        # Run evaluation
        true_labels, predictions, prediction_scores, test_df = evaluator.evaluate_model(args.test_data)
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics(true_labels, predictions)
        
        # Generate report
        evaluator.generate_report(metrics, prediction_scores, args.output_dir)
        
        # Print summary
        print(f"\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Total samples: {len(true_labels)}")
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.1%}")
        print(f"Weighted F1-Score: {metrics['f1_weighted']:.3f}")
        print("\nTop performing classes:")
        
        # Sort classes by F1 score
        class_f1 = [(name, data['f1_score']) for name, data in metrics['per_class_metrics'].items()]
        class_f1.sort(key=lambda x: x[1], reverse=True)
        
        for class_name, f1_score in class_f1:
            print(f"  {class_name}: {f1_score:.3f}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
