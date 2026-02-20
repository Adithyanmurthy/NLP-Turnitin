"""
Script to build MinHash/LSH reference index from corpus
"""

import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reference_index import ReferenceIndexBuilder


def main():
    parser = argparse.ArgumentParser(description="Build MinHash/LSH reference index")
    parser.add_argument("--corpus_path", type=str, required=True,
                        help="Path to corpus directory containing reference documents")
    parser.add_argument("--output_path", type=str, default="../reference_index",
                        help="Path to save the index")
    parser.add_argument("--file_pattern", type=str, default="*.txt",
                        help="File pattern to match (e.g., '*.txt', '*.json')")
    parser.add_argument("--num_perm", type=int, default=128,
                        help="Number of permutations for MinHash (higher = more accurate)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Jaccard similarity threshold for LSH (0.0 to 1.0)")
    parser.add_argument("--shingle_size", type=int, default=3,
                        help="Size of word shingles (k-grams)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.corpus_path):
        print(f"Error: Corpus path does not exist: {args.corpus_path}")
        sys.exit(1)
    
    if args.threshold < 0.0 or args.threshold > 1.0:
        print(f"Error: Threshold must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Create index builder
    print("="*60)
    print("Building MinHash/LSH Reference Index")
    print("="*60)
    print(f"Corpus path: {args.corpus_path}")
    print(f"Output path: {args.output_path}")
    print(f"File pattern: {args.file_pattern}")
    print(f"Num permutations: {args.num_perm}")
    print(f"Threshold: {args.threshold}")
    print(f"Shingle size: {args.shingle_size}")
    print("="*60)
    
    builder = ReferenceIndexBuilder(
        num_perm=args.num_perm,
        threshold=args.threshold,
        shingle_size=args.shingle_size
    )
    
    # Build index
    builder.build_index(
        corpus_path=args.corpus_path,
        output_path=args.output_path,
        file_pattern=args.file_pattern
    )
    
    print("\n" + "="*60)
    print("Index building complete!")
    print("="*60)


if __name__ == "__main__":
    main()
