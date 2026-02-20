"""
Full System Benchmark Script
Person 4: Benchmark the complete integrated system

This script tests the performance of the entire pipeline with various text samples.
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Any
import statistics

from src.pipeline import ContentIntegrityPipeline
from src.config import PipelineConfig
from src.utils import setup_logging, get_logger

setup_logging("INFO")
logger = get_logger(__name__)


# Test samples of varying lengths
TEST_SAMPLES = [
    {
        "name": "Short Text (100 words)",
        "text": """
        Machine learning is a subset of artificial intelligence that enables systems
        to learn and improve from experience without being explicitly programmed.
        It focuses on the development of computer programs that can access data and
        use it to learn for themselves. The process of learning begins with observations
        or data, such as examples, direct experience, or instruction, in order to look
        for patterns in data and make better decisions in the future based on the examples
        that we provide. The primary aim is to allow the computers to learn automatically
        without human intervention or assistance and adjust actions accordingly.
        """ * 1
    },
    {
        "name": "Medium Text (500 words)",
        "text": """
        Artificial intelligence has revolutionized numerous industries and continues to
        shape the future of technology. From healthcare to finance, AI applications are
        becoming increasingly sophisticated and widespread. Machine learning algorithms
        can now process vast amounts of data, identify patterns, and make predictions
        with remarkable accuracy. Deep learning, a subset of machine learning, has
        enabled breakthroughs in computer vision, natural language processing, and
        speech recognition. These advancements have led to the development of virtual
        assistants, autonomous vehicles, and advanced medical diagnostic tools.
        """ * 5
    },
    {
        "name": "Long Text (1000 words)",
        "text": """
        The field of natural language processing has seen tremendous growth in recent years,
        driven by advances in deep learning and the availability of large-scale datasets.
        Modern NLP systems can understand context, sentiment, and even generate human-like
        text. Transformer architectures, introduced in 2017, have become the foundation
        for state-of-the-art language models. These models can perform a wide range of
        tasks, from translation and summarization to question answering and text generation.
        The implications of these technologies are far-reaching, affecting how we interact
        with computers, consume information, and communicate with each other.
        """ * 10
    }
]


def benchmark_module(pipeline: ContentIntegrityPipeline, text: str, module_name: str) -> Dict[str, Any]:
    """
    Benchmark a single module
    
    Args:
        pipeline: Pipeline instance
        text: Test text
        module_name: Name of module to benchmark
    
    Returns:
        Benchmark results dictionary
    """
    times = []
    
    # Warm-up run
    if module_name == "ai_detection":
        pipeline.detect_ai(text, use_cache=False)
    elif module_name == "plagiarism":
        pipeline.check_plagiarism(text, use_cache=False)
    elif module_name == "humanization":
        pipeline.humanize(text, use_cache=False)
    
    # Benchmark runs
    num_runs = 5
    for i in range(num_runs):
        start_time = time.time()
        
        if module_name == "ai_detection":
            result = pipeline.detect_ai(text, use_cache=False)
        elif module_name == "plagiarism":
            result = pipeline.check_plagiarism(text, use_cache=False)
        elif module_name == "humanization":
            result = pipeline.humanize(text, use_cache=False)
        
        elapsed = time.time() - start_time
        times.append(elapsed)
        logger.info(f"  Run {i+1}/{num_runs}: {elapsed:.3f}s")
    
    return {
        "module": module_name,
        "num_runs": num_runs,
        "mean_time": statistics.mean(times),
        "median_time": statistics.median(times),
        "min_time": min(times),
        "max_time": max(times),
        "std_dev": statistics.stdev(times) if len(times) > 1 else 0
    }


def benchmark_full_pipeline(pipeline: ContentIntegrityPipeline, text: str) -> Dict[str, Any]:
    """
    Benchmark the complete pipeline
    
    Args:
        pipeline: Pipeline instance
        text: Test text
    
    Returns:
        Benchmark results dictionary
    """
    times = []
    
    # Warm-up
    pipeline.analyze(text, check_ai=True, check_plagiarism=True, humanize=True, use_cache=False)
    
    # Benchmark runs
    num_runs = 3
    for i in range(num_runs):
        start_time = time.time()
        result = pipeline.analyze(
            text,
            check_ai=True,
            check_plagiarism=True,
            humanize=True,
            use_cache=False
        )
        elapsed = time.time() - start_time
        times.append(elapsed)
        logger.info(f"  Run {i+1}/{num_runs}: {elapsed:.3f}s")
    
    return {
        "module": "full_pipeline",
        "num_runs": num_runs,
        "mean_time": statistics.mean(times),
        "median_time": statistics.median(times),
        "min_time": min(times),
        "max_time": max(times),
        "std_dev": statistics.stdev(times) if len(times) > 1 else 0
    }


def run_benchmarks():
    """Run all benchmarks"""
    logger.info("=" * 70)
    logger.info("CONTENT INTEGRITY PLATFORM - FULL SYSTEM BENCHMARK")
    logger.info("=" * 70)
    
    # Initialize pipeline
    logger.info("\nInitializing pipeline...")
    config = PipelineConfig()
    pipeline = ContentIntegrityPipeline(config)
    
    # Check module availability
    health = pipeline.health_check()
    logger.info(f"\nModule Status:")
    logger.info(f"  AI Detector: {'✓' if health['ai_detector'] else '✗'}")
    logger.info(f"  Plagiarism Detector: {'✓' if health['plagiarism_detector'] else '✗'}")
    logger.info(f"  Humanizer: {'✓' if health['humanizer'] else '✗'}")
    
    all_results = []
    
    # Run benchmarks for each sample
    for sample in TEST_SAMPLES:
        logger.info("\n" + "=" * 70)
        logger.info(f"Benchmarking: {sample['name']}")
        logger.info(f"Text length: {len(sample['text'])} characters")
        logger.info("=" * 70)
        
        sample_results = {
            "sample_name": sample['name'],
            "text_length": len(sample['text']),
            "modules": []
        }
        
        # Benchmark AI Detection
        logger.info("\n[1/4] AI Detection Module")
        ai_results = benchmark_module(pipeline, sample['text'], "ai_detection")
        sample_results['modules'].append(ai_results)
        logger.info(f"  Mean: {ai_results['mean_time']:.3f}s ± {ai_results['std_dev']:.3f}s")
        
        # Benchmark Plagiarism Detection
        logger.info("\n[2/4] Plagiarism Detection Module")
        plag_results = benchmark_module(pipeline, sample['text'], "plagiarism")
        sample_results['modules'].append(plag_results)
        logger.info(f"  Mean: {plag_results['mean_time']:.3f}s ± {plag_results['std_dev']:.3f}s")
        
        # Benchmark Humanization
        logger.info("\n[3/4] Humanization Module")
        human_results = benchmark_module(pipeline, sample['text'], "humanization")
        sample_results['modules'].append(human_results)
        logger.info(f"  Mean: {human_results['mean_time']:.3f}s ± {human_results['std_dev']:.3f}s")
        
        # Benchmark Full Pipeline
        logger.info("\n[4/4] Full Pipeline (All Modules)")
        full_results = benchmark_full_pipeline(pipeline, sample['text'])
        sample_results['modules'].append(full_results)
        logger.info(f"  Mean: {full_results['mean_time']:.3f}s ± {full_results['std_dev']:.3f}s")
        
        all_results.append(sample_results)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 70)
    
    for result in all_results:
        logger.info(f"\n{result['sample_name']} ({result['text_length']} chars):")
        for module in result['modules']:
            logger.info(f"  {module['module']:20s}: {module['mean_time']:.3f}s (±{module['std_dev']:.3f}s)")
    
    # Save results
    output_file = Path("benchmarks/results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")
    logger.info("\nBenchmark complete!")


if __name__ == "__main__":
    run_benchmarks()
