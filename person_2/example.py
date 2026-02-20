"""
Simple example demonstrating plagiarism detection
"""

import os
from src.reference_index import ReferenceIndexBuilder, ReferenceIndexQuery
from src.plagiarism_detector import PlagiarismDetector


def example_build_index():
    """Example: Build a simple reference index."""
    print("="*60)
    print("Example 1: Building Reference Index")
    print("="*60)
    
    # Sample reference documents
    reference_docs = [
        {
            'id': 'doc1',
            'text': '''
            Machine learning is a subset of artificial intelligence that focuses on 
            the development of algorithms and statistical models. These models enable 
            computers to perform tasks without explicit instructions, relying instead 
            on patterns and inference.
            '''
        },
        {
            'id': 'doc2',
            'text': '''
            Deep learning is a type of machine learning based on artificial neural 
            networks. The learning process is deep because the structure of artificial 
            neural networks consists of multiple input, output, and hidden layers.
            '''
        },
        {
            'id': 'doc3',
            'text': '''
            Natural language processing is a branch of artificial intelligence that 
            helps computers understand, interpret and manipulate human language. 
            NLP draws from many disciplines, including computer science and linguistics.
            '''
        },
        {
            'id': 'doc4',
            'text': '''
            The quick brown fox jumps over the lazy dog. This pangram contains 
            every letter of the English alphabet at least once. It is commonly 
            used for testing typewriters and computer keyboards.
            '''
        }
    ]
    
    # Build index
    builder = ReferenceIndexBuilder(
        num_perm=128,
        threshold=0.4,
        shingle_size=3
    )
    
    print(f"\nBuilding index from {len(reference_docs)} documents...")
    builder.build_from_dataset(reference_docs, output_path="example_index")
    print("Index built successfully!")
    
    return "example_index"


def example_query_index(index_path):
    """Example: Query the index for similar documents."""
    print("\n" + "="*60)
    print("Example 2: Querying the Index")
    print("="*60)
    
    # Load index
    index = ReferenceIndexQuery(index_path)
    
    # Query with text similar to doc1
    query_text = """
    Machine learning is a branch of AI that focuses on developing algorithms 
    and models. These models allow computers to perform tasks without being 
    explicitly programmed, using patterns instead.
    """
    
    print(f"\nQuery text: {query_text.strip()[:100]}...")
    print("\nSearching for similar documents...")
    
    results = index.query(query_text, top_k=3)
    
    print(f"\nFound {len(results)} candidate matches:")
    for doc_id, similarity in results:
        print(f"  - {doc_id}: Jaccard similarity = {similarity:.3f}")


def example_detect_plagiarism(index_path):
    """Example: Full plagiarism detection pipeline."""
    print("\n" + "="*60)
    print("Example 3: Plagiarism Detection")
    print("="*60)
    
    # Initialize detector (using pretrained models, no fine-tuning needed for demo)
    print("\nInitializing plagiarism detector...")
    detector = PlagiarismDetector(
        index_path=index_path,
        models_path=None,  # Use pretrained models
        use_sbert=True,
        use_cross_encoder=True
    )
    
    # Test case 1: High plagiarism (paraphrased from doc1)
    print("\n" + "-"*60)
    print("Test Case 1: Paraphrased Text (Expected: High Plagiarism)")
    print("-"*60)
    
    test_text_1 = """
    Machine learning is a subset of AI that concentrates on creating algorithms 
    and statistical models. These models enable computers to complete tasks 
    without explicit programming, depending on patterns and inference instead.
    """
    
    print(f"Input: {test_text_1.strip()[:100]}...")
    report_1 = detector.check(test_text_1, top_k_candidates=3)
    
    print(f"\nResults:")
    print(f"  Score: {report_1['score']:.2%}")
    print(f"  Verdict: {report_1['verdict']}")
    print(f"  Matches: {report_1['num_matches']}")
    
    if report_1['matches']:
        top_match = report_1['matches'][0]
        print(f"\n  Top Match:")
        print(f"    Source: {top_match['source']}")
        print(f"    Similarity: {top_match['similarity']:.2%}")
    
    # Test case 2: Original text (Expected: Low/No Plagiarism)
    print("\n" + "-"*60)
    print("Test Case 2: Original Text (Expected: Low/No Plagiarism)")
    print("-"*60)
    
    test_text_2 = """
    Quantum computing represents a revolutionary approach to information processing.
    Unlike classical computers that use bits, quantum computers use quantum bits or
    qubits, which can exist in multiple states simultaneously through superposition.
    """
    
    print(f"Input: {test_text_2.strip()[:100]}...")
    report_2 = detector.check(test_text_2, top_k_candidates=3)
    
    print(f"\nResults:")
    print(f"  Score: {report_2['score']:.2%}")
    print(f"  Verdict: {report_2['verdict']}")
    print(f"  Matches: {report_2['num_matches']}")
    
    # Test case 3: Direct copy (Expected: Very High Plagiarism)
    print("\n" + "-"*60)
    print("Test Case 3: Direct Copy (Expected: Very High Plagiarism)")
    print("-"*60)
    
    test_text_3 = """
    The quick brown fox jumps over the lazy dog. This pangram contains 
    every letter of the English alphabet at least once.
    """
    
    print(f"Input: {test_text_3.strip()[:100]}...")
    report_3 = detector.check(test_text_3, top_k_candidates=3)
    
    print(f"\nResults:")
    print(f"  Score: {report_3['score']:.2%}")
    print(f"  Verdict: {report_3['verdict']}")
    print(f"  Matches: {report_3['num_matches']}")
    
    if report_3['matches']:
        top_match = report_3['matches'][0]
        print(f"\n  Top Match:")
        print(f"    Source: {top_match['source']}")
        print(f"    Similarity: {top_match['similarity']:.2%}")
        if top_match['sentences']:
            print(f"\n  Matched Sentences:")
            for i, sent in enumerate(top_match['sentences'][:2], 1):
                print(f"    {i}. Score: {sent['score']:.2%}")
                print(f"       Input: {sent['input_sentence'][:60]}...")
                print(f"       Match: {sent['matched_sentence'][:60]}...")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("PLAGIARISM DETECTION ENGINE - EXAMPLES")
    print("="*60)
    
    # Example 1: Build index
    index_path = example_build_index()
    
    # Example 2: Query index
    example_query_index(index_path)
    
    # Example 3: Full plagiarism detection
    example_detect_plagiarism(index_path)
    
    print("\n" + "="*60)
    print("Examples completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Build a larger index from your reference corpus")
    print("2. Fine-tune models on domain-specific data")
    print("3. Integrate with Person 4's pipeline module")
    print("\nSee USAGE.md for detailed documentation.")


if __name__ == "__main__":
    main()
