"""
Unit tests for MinHash/LSH reference index
"""

import os
import sys
import tempfile
import shutil
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reference_index import ReferenceIndexBuilder, ReferenceIndexQuery


class TestReferenceIndex:
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            {
                'id': 'doc1',
                'text': 'The quick brown fox jumps over the lazy dog. This is a test document.'
            },
            {
                'id': 'doc2',
                'text': 'The quick brown fox leaps over the sleeping dog. This is another test.'
            },
            {
                'id': 'doc3',
                'text': 'Machine learning is a subset of artificial intelligence. It focuses on data.'
            }
        ]
    
    @pytest.fixture
    def temp_index_path(self):
        """Create temporary directory for index."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_index_builder_creation(self):
        """Test creating an index builder."""
        builder = ReferenceIndexBuilder(num_perm=128, threshold=0.5, shingle_size=3)
        assert builder.num_perm == 128
        assert builder.threshold == 0.5
        assert builder.shingle_size == 3
    
    def test_add_document(self, sample_documents):
        """Test adding documents to index."""
        builder = ReferenceIndexBuilder()
        
        for doc in sample_documents:
            builder.add_document(doc['id'], doc['text'])
        
        assert len(builder.documents) == 3
        assert len(builder.minhashes) == 3
        assert 'doc1' in builder.documents
    
    def test_build_from_dataset(self, sample_documents, temp_index_path):
        """Test building index from dataset."""
        builder = ReferenceIndexBuilder()
        builder.build_from_dataset(sample_documents, temp_index_path)
        
        # Check that index files were created
        assert os.path.exists(os.path.join(temp_index_path, 'lsh_index.pkl'))
        assert os.path.exists(os.path.join(temp_index_path, 'documents.json'))
        assert os.path.exists(os.path.join(temp_index_path, 'minhashes.pkl'))
        assert os.path.exists(os.path.join(temp_index_path, 'metadata.json'))
    
    def test_query_index(self, sample_documents, temp_index_path):
        """Test querying the index."""
        # Build index
        builder = ReferenceIndexBuilder(threshold=0.3)
        builder.build_from_dataset(sample_documents, temp_index_path)
        
        # Load index
        query_engine = ReferenceIndexQuery(temp_index_path)
        
        # Query with similar text to doc1
        query_text = "The quick brown fox jumps over the lazy dog."
        results = query_engine.query(query_text, top_k=5)
        
        # Should find doc1 and doc2 (similar texts)
        assert len(results) > 0
        assert results[0][0] in ['doc1', 'doc2']  # Top result should be doc1 or doc2
        assert results[0][1] > 0.3  # Jaccard similarity should be > threshold
    
    def test_get_document(self, sample_documents, temp_index_path):
        """Test retrieving document by ID."""
        builder = ReferenceIndexBuilder()
        builder.build_from_dataset(sample_documents, temp_index_path)
        
        query_engine = ReferenceIndexQuery(temp_index_path)
        
        doc_text = query_engine.get_document('doc1')
        assert doc_text is not None
        assert 'quick brown fox' in doc_text
    
    def test_get_statistics(self, sample_documents, temp_index_path):
        """Test getting index statistics."""
        builder = ReferenceIndexBuilder()
        builder.build_from_dataset(sample_documents, temp_index_path)
        
        query_engine = ReferenceIndexQuery(temp_index_path)
        stats = query_engine.get_statistics()
        
        assert stats['num_documents'] == 3
        assert stats['num_perm'] == 128
        assert stats['threshold'] == 0.5
        assert 'index_size_mb' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
