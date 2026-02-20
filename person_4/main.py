#!/usr/bin/env python3
"""
Content Integrity Platform - Command Line Interface
Person 4: CLI entry point

Usage:
    python main.py --input "your text here"
    python main.py --file document.pdf          # PDF, DOCX, TXT supported
    python main.py --input "text" --detect --plagiarism
    python main.py --input "text" --humanize
    python main.py --input "text" --deplagiarize
    python main.py --input "text" --full
    echo "text" | python main.py --stdin
"""

import sys
import argparse
from pathlib import Path

from src.pipeline import ContentIntegrityPipeline
from src.config import load_config
from src.utils import setup_logging, get_logger, format_report, save_json


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Content Integrity & Authorship Intelligence Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze text directly
  python main.py --input "Your text here" --full
  
  # Analyze from file (TXT, PDF, DOCX supported)
  python main.py --file document.pdf --detect --plagiarism
  python main.py --file essay.docx --full
  
  # Humanize AI-generated text (target: ≤5% AI score)
  python main.py --input "AI text" --humanize
  
  # Deplagiarize text (rewrite plagiarized sections to ≤5%)
  python main.py --input "text" --deplagiarize
  
  # Full pipeline: detect + plagiarism + humanize + deplagiarize
  python main.py --file report.pdf --full
  
  # Read from stdin
  echo "Text to analyze" | python main.py --stdin --full
  
  # Save output to file
  python main.py --input "text" --full --output report.json
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input', '-i',
        type=str,
        help='Input text as string argument'
    )
    input_group.add_argument(
        '--file', '-f',
        type=str,
        help='Path to input text file'
    )
    input_group.add_argument(
        '--stdin',
        action='store_true',
        help='Read input from stdin'
    )
    
    # Analysis options
    parser.add_argument(
        '--detect', '--ai',
        action='store_true',
        dest='detect_ai',
        help='Run AI detection'
    )
    parser.add_argument(
        '--plagiarism', '--plag',
        action='store_true',
        help='Check for plagiarism'
    )
    parser.add_argument(
        '--humanize', '--human',
        action='store_true',
        help='Humanize the text (transform AI to human-like, target ≤5%%)'
    )
    parser.add_argument(
        '--deplagiarize', '--deplag',
        action='store_true',
        help='Deplagiarize the text (rewrite plagiarized sections to ≤5%%)'
    )
    parser.add_argument(
        '--full', '--all',
        action='store_true',
        help='Run all analyses (AI detection + plagiarism + humanization + deplagiarization)'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Save output to file (JSON format)'
    )
    parser.add_argument(
        '--format',
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (YAML)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output (DEBUG level logging)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode (only show results)'
    )
    
    return parser.parse_args()


def read_input(args) -> str:
    """
    Read input text from specified source.
    For --file, uses the file_parser to handle PDF, DOCX, TXT, HTML.
    """
    if args.input:
        return args.input
    
    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        
        from src.file_parser import parse_file
        try:
            text, fmt = parse_file(str(file_path))
            print(f"Parsed {fmt} file: {file_path.name} ({len(text)} characters)")
            return text
        except Exception as e:
            print(f"Error parsing file: {e}", file=sys.stderr)
            sys.exit(1)
    
    elif args.stdin:
        return sys.stdin.read()
    
    else:
        print("Error: No input provided", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point"""
    args = parse_arguments()
    
    # Setup logging
    if args.quiet:
        log_level = "ERROR"
    elif args.verbose:
        log_level = "DEBUG"
    else:
        log_level = "INFO"
    
    setup_logging(log_level)
    logger = get_logger(__name__)
    
    # Load configuration
    config = load_config(args.config) if args.config else None
    
    # Initialize pipeline
    try:
        logger.info("Initializing Content Integrity Pipeline...")
        pipeline = ContentIntegrityPipeline(config)
    except Exception as e:
        print(f"Error initializing pipeline: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Read input
    try:
        text = read_input(args)
        logger.info(f"Input text length: {len(text)} characters")
    except Exception as e:
        print(f"Error reading input: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Determine which analyses to run
    if args.full:
        check_ai = True
        check_plagiarism = True
        humanize = True
        deplagiarize = True
    else:
        check_ai = args.detect_ai
        check_plagiarism = args.plagiarism
        humanize = args.humanize
        deplagiarize = args.deplagiarize
        
        # If no specific analysis selected, default to AI detection
        if not (check_ai or check_plagiarism or humanize or deplagiarize):
            check_ai = True
    
    # Run analysis
    try:
        logger.info("Starting analysis...")
        report = pipeline.analyze(
            text=text,
            check_ai=check_ai,
            check_plagiarism=check_plagiarism,
            humanize=humanize,
            deplagiarize=deplagiarize,
            use_cache=not args.no_cache
        )
        logger.info("Analysis complete")
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        logger.exception("Analysis failed")
        sys.exit(1)
    
    # Format output
    if args.format == 'json':
        output = format_report(report, 'json')
    else:
        output = format_report(report, 'text')
    
    # Display output
    if not args.quiet:
        print()  # Blank line before output
    print(output)
    
    # Save to file if requested
    if args.output:
        try:
            output_path = Path(args.output)
            if args.format == 'json':
                save_json(report, output_path)
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(output)
            logger.info(f"Output saved to: {args.output}")
        except Exception as e:
            print(f"Error saving output: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Exit with appropriate code
    sys.exit(0)


if __name__ == '__main__':
    main()
