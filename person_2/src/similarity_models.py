"""
Semantic Similarity Models for Plagiarism Detection
Wrappers for Sentence-BERT, SimCSE, Cross-Encoder, and Longformer
"""

import torch
import numpy as np
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModel, LongformerModel, LongformerTokenizer
import torch.nn.functional as F


class SentenceBERTModel:
    """
    Sentence-BERT model for sentence embeddings.
    Uses all-mpnet-base-v2 as the base model.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: Optional[str] = None
    ):
        """
        Initialize Sentence-BERT model.
        
        Args:
            model_name: HuggingFace model name or path to fine-tuned checkpoint
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model_name = model_name
        
        print(f"Loaded Sentence-BERT model: {model_name} on {self.device}")
    
    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode sentences into embeddings.
        
        Args:
            sentences: Single sentence or list of sentences
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize: Normalize embeddings to unit length
        
        Returns:
            Numpy array of embeddings (shape: [num_sentences, embedding_dim])
        """
        embeddings = self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        
        return embeddings
    
    def compute_similarity(
        self,
        sentences1: Union[str, List[str]],
        sentences2: Union[str, List[str]]
    ) -> np.ndarray:
        """
        Compute cosine similarity between two sets of sentences.
        
        Args:
            sentences1: First set of sentences
            sentences2: Second set of sentences
        
        Returns:
            Similarity matrix (shape: [len(sentences1), len(sentences2)])
        """
        # Ensure inputs are lists
        if isinstance(sentences1, str):
            sentences1 = [sentences1]
        if isinstance(sentences2, str):
            sentences2 = [sentences2]
        
        # Encode sentences
        embeddings1 = self.encode(sentences1, normalize=True)
        embeddings2 = self.encode(sentences2, normalize=True)
        
        # Compute cosine similarity
        similarity_matrix = np.dot(embeddings1, embeddings2.T)
        
        return similarity_matrix


class SimCSEModel:
    """
    SimCSE model for contrastive sentence embeddings.
    """
    
    def __init__(
        self,
        model_name: str = "princeton-nlp/sup-simcse-roberta-large",
        device: Optional[str] = None
    ):
        """
        Initialize SimCSE model.
        
        Args:
            model_name: HuggingFace model name or path to fine-tuned checkpoint
            device: Device to use
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.model_name = model_name
        
        print(f"Loaded SimCSE model: {model_name} on {self.device}")
    
    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode sentences into embeddings.
        
        Args:
            sentences: Single sentence or list of sentences
            batch_size: Batch size for encoding
            normalize: Normalize embeddings
        
        Returns:
            Numpy array of embeddings
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get embeddings (use [CLS] token)
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
                
                # Normalize if requested
                if normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def compute_similarity(
        self,
        sentences1: Union[str, List[str]],
        sentences2: Union[str, List[str]]
    ) -> np.ndarray:
        """
        Compute cosine similarity between two sets of sentences.
        
        Args:
            sentences1: First set of sentences
            sentences2: Second set of sentences
        
        Returns:
            Similarity matrix
        """
        embeddings1 = self.encode(sentences1, normalize=True)
        embeddings2 = self.encode(sentences2, normalize=True)
        
        similarity_matrix = np.dot(embeddings1, embeddings2.T)
        
        return similarity_matrix


class CrossEncoderModel:
    """
    DeBERTa-v3 Cross-Encoder for pairwise similarity verification.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-large",
        device: Optional[str] = None
    ):
        """
        Initialize Cross-Encoder model.
        
        Args:
            model_name: HuggingFace model name or path to fine-tuned checkpoint
            device: Device to use
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CrossEncoder(model_name, device=self.device)
        self.model_name = model_name
        
        print(f"Loaded Cross-Encoder model: {model_name} on {self.device}")
    
    def predict(
        self,
        sentence_pairs: List[tuple],
        batch_size: int = 16,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Predict similarity scores for sentence pairs.
        
        Args:
            sentence_pairs: List of (sentence1, sentence2) tuples
            batch_size: Batch size for prediction
            show_progress: Show progress bar
        
        Returns:
            Array of similarity scores
        """
        scores = self.model.predict(
            sentence_pairs,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        # Normalize scores to [0, 1] range if needed
        # Cross-encoder scores are typically in [-1, 1] or [0, 1] depending on training
        if scores.min() < 0:
            scores = (scores + 1) / 2  # Map [-1, 1] to [0, 1]
        
        return scores
    
    def verify_pairs(
        self,
        input_sentences: List[str],
        reference_sentences: List[str],
        threshold: float = 0.7
    ) -> List[tuple]:
        """
        Verify which sentence pairs exceed similarity threshold.
        
        Args:
            input_sentences: List of input sentences
            reference_sentences: List of reference sentences
            threshold: Similarity threshold
        
        Returns:
            List of (input_idx, ref_idx, score) tuples for matches
        """
        # Create all pairs
        pairs = []
        pair_indices = []
        
        for i, input_sent in enumerate(input_sentences):
            for j, ref_sent in enumerate(reference_sentences):
                pairs.append((input_sent, ref_sent))
                pair_indices.append((i, j))
        
        # Predict scores
        scores = self.predict(pairs)
        
        # Filter by threshold
        matches = []
        for (i, j), score in zip(pair_indices, scores):
            if score >= threshold:
                matches.append((i, j, float(score)))
        
        return matches


class LongformerSimilarity:
    """
    Longformer model for document-level similarity comparison.
    """
    
    def __init__(
        self,
        model_name: str = "allenai/longformer-base-4096",
        device: Optional[str] = None
    ):
        """
        Initialize Longformer model.
        
        Args:
            model_name: HuggingFace model name or path to fine-tuned checkpoint
            device: Device to use
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = LongformerTokenizer.from_pretrained(model_name)
        self.model = LongformerModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.model_name = model_name
        
        print(f"Loaded Longformer model: {model_name} on {self.device}")
    
    def encode(
        self,
        documents: Union[str, List[str]],
        max_length: int = 4096,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode long documents into embeddings.
        
        Args:
            documents: Single document or list of documents
            max_length: Maximum sequence length (up to 4096)
            normalize: Normalize embeddings
        
        Returns:
            Numpy array of embeddings
        """
        if isinstance(documents, str):
            documents = [documents]
        
        all_embeddings = []
        
        with torch.no_grad():
            for doc in documents:
                # Tokenize
                inputs = self.tokenizer(
                    doc,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get embeddings (mean pooling)
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
                
                # Normalize if requested
                if normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def compute_similarity(
        self,
        documents1: Union[str, List[str]],
        documents2: Union[str, List[str]]
    ) -> np.ndarray:
        """
        Compute cosine similarity between long documents.
        
        Args:
            documents1: First set of documents
            documents2: Second set of documents
        
        Returns:
            Similarity matrix
        """
        embeddings1 = self.encode(documents1, normalize=True)
        embeddings2 = self.encode(documents2, normalize=True)
        
        similarity_matrix = np.dot(embeddings1, embeddings2.T)
        
        return similarity_matrix
