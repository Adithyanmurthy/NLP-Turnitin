"""
Evaluation script for PAN Plagiarism Corpus
Evaluates the plagiarism detector on standard PAN test sets
"""

import os
import sys
import argparse
import json
from typing import List, Dict
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plagiarism_detector import PlagiarismDetector


def load_pan_test_set(test_path: str) -> List[Dict]:
    """
    Load PAN test set.
    
    Expected format: JSON file with list of test cases
    Each case has: 'text', 'label' (0=no plagiarism, 1=plagiarism), 'source' (optional)
    
    Args:
        test_path: Path to test set JSON file
    
    Returns:
        List of test cases
    """
    with open(test_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    return test_cases


def evaluate_detector(
    detector: PlagiarismDetector,
    test_cases: List[Dict],
    threshold: float = 0.5
) -> Dict:
    """
    Evaluate detector on test cases.
    
    Args:
        detector: PlagiarismDetector instance
        test_cases: List of test cases
        threshold: Score threshold for binary classification
    
    Returns:
        Dictionary with evaluation metrics
    """
    y_true = []
    y_pred = []
    y_scores = []
    
    print(f"Evaluating on {len(test_cases)} test cases...")
    
    for i, case in enumerate(test_cases):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(test_cases)}")
        
        text = case['text']
        true_label = case['label']
        
        # Run detection
        try:
            report = detector.check(text, top_k_candidates=5)
            score = report['score']
            pred_label = 1 if score >= threshold else 0
        except Exception as e:
            print(f"Error on case {i}: {e}")
            score = 0.0
            pred_label = 0
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        y_scores.append(score)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate AUROC if possible
    try:
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(y_true, y_scores)
    except:
        auroc = None
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc,
        'threshold': threshold,
        'num_test_cases': len(test_cases),
        'predictions': {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_scores': y_scores
        }
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate plagiarism detector on PAN corpus")
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path to PAN test set JSON file")
    parser.add_argument("--index_path", type=str, default="../reference_index",
                        help="Path to MinHash/LSH index")
    parser.add_argument("--models_path", type=str, default="../checkpoints",
                        help="Path to trained model checkpoints")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Score threshold for binary classification")
    parser.add_argument("--output", type=str,
                        help="Path to save evaluation results (JSON)")
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.test_path):
        print(f"Error: Test set not found: {args.test_path}")
        sys.exit(1)
    
    if not os.path.exists(args.index_path):
        print(f"Error: Index not found: {args.index_path}")
        sys.exit(1)
    
    # Load test set
    print("="*60)
    print("Loading test set...")
    test_cases = load_pan_test_set(args.test_path)
    print(f"Loaded {len(test_cases)} test cases")
    
    # Initialize detector
    print("\nInitializing detector...")
    detector = PlagiarismDetector(
        index_path=args.index_path,
        models_path=args.models_path if os.path.exists(args.models_path) else None,
        use_sbert=True,
        use_cross_encoder=True
    )
    
    # Run evaluation
    print("\n" + "="*60)
    print("Running evaluation...")
    print("="*60)
    
    results = evaluate_detector(detector, test_cases, threshold=args.threshold)
    
    # Display results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Test cases: {results['num_test_cases']}")
    print(f"Threshold: {results['threshold']}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1 Score:  {results['f1']:.4f}")
    if results['auroc'] is not None:
        print(f"  AUROC:     {results['auroc']:.4f}")
    print("="*60)
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
