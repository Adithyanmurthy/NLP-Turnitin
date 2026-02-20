"""
Person 3 - Example Usage
Demonstrates how to use the humanization module
"""

def example_basic_usage():
    """Basic usage example"""
    print("=" * 80)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 80)
    
    from humanizer import humanize
    
    # AI-generated text
    ai_text = """
    Artificial intelligence represents a transformative technological paradigm 
    that has fundamentally revolutionized numerous industrial sectors through 
    the implementation of sophisticated algorithmic frameworks and computational 
    methodologies. Machine learning systems demonstrate remarkable capabilities 
    in processing extensive datasets and identifying complex patterns that may 
    elude human cognitive analysis.
    """
    
    print("\nInput (AI-generated):")
    print(ai_text)
    
    # Humanize
    result = humanize(ai_text)
    
    print("\nOutput (Humanized):")
    print(result["text"])
    
    print("\nMetrics:")
    print(f"  AI Score Before: {result['ai_score_before']:.2%}")
    print(f"  AI Score After:  {result['ai_score_after']:.2%}")
    print(f"  Iterations:      {result['iterations']}")
    print(f"  Diversity Used:  {result['diversity_used']}")
    print(f"  Reorder Used:    {result['reorder_used']}")

def example_advanced_usage():
    """Advanced usage with custom model"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Advanced Usage with Custom Model")
    print("=" * 80)
    
    from humanizer import Humanizer
    
    # Initialize with specific model
    humanizer = Humanizer(model_name="pegasus", use_feedback=True)
    
    text = "The implementation of neural networks facilitates automated decision-making processes."
    
    print(f"\nInput: {text}")
    
    # Humanize with custom parameters
    result = humanizer.paraphrase(text, diversity=80, reorder=60)
    
    print(f"\nOutput: {result}")

def example_batch_processing():
    """Batch processing example"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Batch Processing")
    print("=" * 80)
    
    from humanizer import Humanizer
    
    humanizer = Humanizer(model_name="flan_t5", use_feedback=False)
    
    texts = [
        "Machine learning algorithms analyze data patterns.",
        "Natural language processing enables text understanding.",
        "Deep learning models require substantial computational resources."
    ]
    
    print("\nProcessing multiple texts:")
    for i, text in enumerate(texts, 1):
        result = humanizer.humanize(text)
        print(f"\n{i}. Input:  {text}")
        print(f"   Output: {result['text']}")

def example_evaluation():
    """Evaluation example"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Evaluation Metrics")
    print("=" * 80)
    
    from humanizer import Humanizer
    from evaluator import HumanizationEvaluator
    
    humanizer = Humanizer(model_name="flan_t5", use_feedback=False)
    evaluator = HumanizationEvaluator()
    
    original = "Artificial intelligence has revolutionized many industries."
    result = humanizer.humanize(original)
    humanized = result["text"]
    
    print(f"\nOriginal:  {original}")
    print(f"Humanized: {humanized}")
    
    # Evaluate
    metrics = evaluator.evaluate(original, humanized)
    
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key:25s}: {value:.4f}")

def main():
    """Run all examples"""
    print("=" * 80)
    print("PERSON 3 - HUMANIZATION MODULE EXAMPLES")
    print("=" * 80)
    
    try:
        example_basic_usage()
        
        response = input("\nRun advanced examples? (yes/no): ")
        if response.lower() in ["yes", "y"]:
            example_advanced_usage()
            example_batch_processing()
            example_evaluation()
        
        print("\n" + "=" * 80)
        print("EXAMPLES COMPLETE")
        print("=" * 80)
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease train models first:")
        print("  python run_all.py")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
