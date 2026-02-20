"""
Script to detect plagiarism in text using the trained models
"""

import os
import sys
import argparse
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plagiarism_detector import PlagiarismDetector


def main():
    parser = argparse.ArgumentParser(description="Detect plagiarism in text")
    parser.add_argument("--text", type=str,
                        help="Text to check for plagiarism")
    parser.add_argument("--file", type=str,
                        help="Path to file containing text to check")
    parser.add_argument("--index_path", type=str, default="../reference_index",
                        help="Path to MinHash/LSH index")
    parser.add_argument("--models_path", type=str, default="../checkpoints",
                        help="Path to trained model checkpoints")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of candidate documents to retrieve")
    parser.add_argument("--sentence_threshold", type=float, default=0.75,
                        help="Similarity threshold for sentence matching")
    parser.add_argument("--verification_threshold", type=float, default=0.7,
                        help="Threshold for cross-encoder verification")
    parser.add_argument("--output", type=str,
                        help="Path to save report (JSON format)")
    parser.add_argument("--no_sbert", action="store_true",
                        help="Disable Sentence-BERT")
    parser.add_argument("--use_simcse", action="store_true",
                        help="Use SimCSE instead of Sentence-BERT")
    parser.add_argument("--no_cross_encoder", action="store_true",
                        help="Disable Cross-Encoder verification")
    parser.add_argument("--use_longformer", action="store_true",
                        help="Enable Longformer for long documents")
    
    args = parser.parse_args()
    
    # Get input text
    if args.text:
        text = args.text
    elif args.file:
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        print("Error: Must provide either --text or --file")
        sys.exit(1)
    
    # Validate index path
    if not os.path.exists(args.index_path):
        print(f"Error: Index path not found: {args.index_path}")
        print("Please build the index first using scripts/build_index.py")
        sys.exit(1)
    
    # Initialize detector
    print("="*60)
    print("Initializing Plagiarism Detector")
    print("="*60)
    
    detector = PlagiarismDetector(
        index_path=args.index_path,
        models_path=args.models_path if os.path.exists(args.models_path) else None,
        use_sbert=not args.no_sbert,
        use_simcse=args.use_simcse,
        use_cross_encoder=not args.no_cross_encoder,
        use_longformer=args.use_longformer
    )
    
    # Run detection
    print("\n" + "="*60)
    print("Running Plagiarism Detection")
    print("="*60)
    print(f"Input text length: {len(text)} characters")
    print(f"Top K candidates: {args.top_k}")
    print(f"Sentence threshold: {args.sentence_threshold}")
    print(f"Verification threshold: {args.verification_threshold}")
    print("="*60 + "\n")
    
    report = detector.check(
        text=text,
        top_k_candidates=args.top_k,
        sentence_threshold=args.sentence_threshold,
        verification_threshold=args.verification_threshold
    )
    
    # Display report
    print("\n" + "="*60)
    print("PLAGIARISM DETECTION REPORT")
    print("="*60)
    print(f"Overall Score: {report['score']:.2%}")
    print(f"Verdict: {report['verdict']}")
    print(f"Number of Matches: {report['num_matches']}")
    print("="*60)
    
    if report['matches']:
        print("\nDetailed Matches:")
        print("-"*60)
        for i, match in enumerate(report['matches'], 1):
            print(f"\nMatch {i}:")
            print(f"  Source: {match['source']}")
            print(f"  Similarity: {match['similarity']:.2%}")
            print(f"  Matched Sentences: {match['num_matched_sentences']}")
            
            if match['sentences']:
                print(f"\n  Top Sentence Matches:")
                for j, sent_match in enumerate(match['sentences'][:3], 1):
                    print(f"\n    {j}. Score: {sent_match['score']:.2%}")
                    print(f"       Input: {sent_match['input_sentence'][:100]}...")
                    print(f"       Match: {sent_match['matched_sentence'][:100]}...")
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to: {args.output}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
