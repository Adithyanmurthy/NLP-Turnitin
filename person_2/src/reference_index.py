"""
MinHash/LSH Reference Index Implementation
Fast approximate duplicate detection for plagiarism screening
"""

import os
import json
import pickle
from typing import List, Dict, Tuple, Optional
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
import numpy as np

from .utils import preprocess_text, create_shingles


class ReferenceIndexBuilder:
    """
    Builds MinHash/LSH index from reference corpus for fast plagiarism screening.
    """
    
    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.5,
        shingle_size: int = 3
    ):
        """
        Initialize the reference index builder.
        
        Args:
            num_perm: Number of permutations for MinHash (higher = more accurate, slower)
            threshold: Jaccard similarity threshold for LSH (0.0 to 1.0)
            shingle_size: Size of word shingles (k-grams)
        """
        self.num_perm = num_perm
        self.threshold = threshold
        self.shingle_size = shingle_size
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.documents = {}  # doc_id -> document text
        self.minhashes = {}  # doc_id -> MinHash object
        
    def _create_minhash(self, text: str) -> MinHash:
        """
        Create MinHash signature for a document.
        
        Args:
            text: Document text
        
        Returns:
            MinHash object
        """
        # Preprocess text
        text = preprocess_text(text, lowercase=True)
        
        # Create shingles
        shingles = create_shingles(text, k=self.shingle_size)
        
        # Create MinHash
        minhash = MinHash(num_perm=self.num_perm)
        for shingle in shingles:
            minhash.update(shingle.encode('utf-8'))
        
        return minhash
    
    def add_document(self, doc_id: str, text: str):
        """
        Add a document to the index.
        
        Args:
            doc_id: Unique document identifier
            text: Document text
        """
        # Create MinHash
        minhash = self._create_minhash(text)
        
        # Store document and MinHash
        self.documents[doc_id] = text
        self.minhashes[doc_id] = minhash
        
        # Insert into LSH index
        self.lsh.insert(doc_id, minhash)
    
    def build_index(
        self,
        corpus_path: str,
        output_path: str,
        file_pattern: str = "*.txt"
    ):
        """
        Build index from a corpus directory.
        
        Args:
            corpus_path: Path to corpus directory
            output_path: Path to save the index
            file_pattern: File pattern to match (e.g., "*.txt")
        """
        print(f"Building reference index from {corpus_path}...")
        
        # Find all documents
        import glob
        doc_files = glob.glob(os.path.join(corpus_path, "**", file_pattern), recursive=True)
        
        if not doc_files:
            print(f"Warning: No files found matching pattern {file_pattern} in {corpus_path}")
            return
        
        print(f"Found {len(doc_files)} documents")
        
        # Process each document
        for doc_file in tqdm(doc_files, desc="Indexing documents"):
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Use relative path as doc_id
                doc_id = os.path.relpath(doc_file, corpus_path)
                self.add_document(doc_id, text)
                
            except Exception as e:
                print(f"Error processing {doc_file}: {e}")
                continue
        
        # Save index
        self.save(output_path)
        print(f"Index saved to {output_path}")
        print(f"Total documents indexed: {len(self.documents)}")
    
    def build_from_dataset(self, documents: List[Dict[str, str]], output_path: str):
        """
        Build index from a list of document dictionaries.
        
        Args:
            documents: List of dicts with 'id' and 'text' keys
            output_path: Path to save the index
        """
        print(f"Building reference index from {len(documents)} documents...")
        
        for doc in tqdm(documents, desc="Indexing documents"):
            doc_id = doc.get('id', str(hash(doc['text'])))
            text = doc['text']
            self.add_document(doc_id, text)
        
        self.save(output_path)
        print(f"Index saved to {output_path}")
    
    def save(self, output_path: str):
        """
        Save the index to disk.
        
        Args:
            output_path: Directory to save index files
        """
        os.makedirs(output_path, exist_ok=True)
        
        # Save LSH index
        with open(os.path.join(output_path, 'lsh_index.pkl'), 'wb') as f:
            pickle.dump(self.lsh, f)
        
        # Save documents
        with open(os.path.join(output_path, 'documents.json'), 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        
        # Save MinHashes
        with open(os.path.join(output_path, 'minhashes.pkl'), 'wb') as f:
            pickle.dump(self.minhashes, f)
        
        # Save metadata
        metadata = {
            'num_perm': self.num_perm,
            'threshold': self.threshold,
            'shingle_size': self.shingle_size,
            'num_documents': len(self.documents)
        }
        with open(os.path.join(output_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)


class ReferenceIndexQuery:
    """
    Query MinHash/LSH index for candidate plagiarism matches.
    """
    
    def __init__(self, index_path: str):
        """
        Load a pre-built index.
        
        Args:
            index_path: Path to index directory
        """
        self.index_path = index_path
        
        # Load metadata
        with open(os.path.join(index_path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        self.num_perm = metadata['num_perm']
        self.threshold = metadata['threshold']
        self.shingle_size = metadata['shingle_size']
        
        # Load LSH index
        with open(os.path.join(index_path, 'lsh_index.pkl'), 'rb') as f:
            self.lsh = pickle.load(f)
        
        # Load documents
        with open(os.path.join(index_path, 'documents.json'), 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        
        # Load MinHashes
        with open(os.path.join(index_path, 'minhashes.pkl'), 'rb') as f:
            self.minhashes = pickle.load(f)
        
        print(f"Loaded index with {len(self.documents)} documents")
    
    def _create_minhash(self, text: str) -> MinHash:
        """
        Create MinHash signature for a query document.
        
        Args:
            text: Document text
        
        Returns:
            MinHash object
        """
        text = preprocess_text(text, lowercase=True)
        shingles = create_shingles(text, k=self.shingle_size)
        
        minhash = MinHash(num_perm=self.num_perm)
        for shingle in shingles:
            minhash.update(shingle.encode('utf-8'))
        
        return minhash
    
    def query(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Query the index for similar documents.
        
        Args:
            text: Query text
            top_k: Maximum number of candidates to return
        
        Returns:
            List of (doc_id, jaccard_similarity) tuples, sorted by similarity
        """
        # Create MinHash for query
        query_minhash = self._create_minhash(text)
        
        # Query LSH index
        candidate_ids = self.lsh.query(query_minhash)
        
        if not candidate_ids:
            return []
        
        # Calculate exact Jaccard similarities
        similarities = []
        for doc_id in candidate_ids:
            if doc_id in self.minhashes:
                ref_minhash = self.minhashes[doc_id]
                jaccard_sim = query_minhash.jaccard(ref_minhash)
                similarities.append((doc_id, jaccard_sim))
        
        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_document(self, doc_id: str) -> Optional[str]:
        """
        Retrieve document text by ID.
        
        Args:
            doc_id: Document identifier
        
        Returns:
            Document text or None if not found
        """
        return self.documents.get(doc_id)
    
    def get_statistics(self) -> Dict:
        """
        Get index statistics.
        
        Returns:
            Dictionary with index statistics
        """
        return {
            'num_documents': len(self.documents),
            'num_perm': self.num_perm,
            'threshold': self.threshold,
            'shingle_size': self.shingle_size,
            'index_size_mb': self._get_index_size()
        }
    
    def _get_index_size(self) -> float:
        """Calculate total size of index files in MB."""
        total_size = 0
        for filename in ['lsh_index.pkl', 'documents.json', 'minhashes.pkl', 'metadata.json']:
            filepath = os.path.join(self.index_path, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
        return total_size / (1024 * 1024)  # Convert to MB
