"""
Integration tests for the main pipeline
Person 4: Test the complete pipeline integration
"""

import pytest
from src.pipeline import ContentIntegrityPipeline
from src.config import PipelineConfig


@pytest.fixture
def pipeline():
    """Create a pipeline instance for testing"""
    config = PipelineConfig()
    return ContentIntegrityPipeline(config)


@pytest.fixture
def sample_text():
    """Sample text for testing"""
    return """
    Machine learning is a subset of artificial intelligence that enables systems
    to learn and improve from experience without being explicitly programmed.
    It focuses on the development of computer programs that can access data and
    use it to learn for themselves.
    """


class TestPipelineInitialization:
    """Test pipeline initialization"""
    
    def test_pipeline_creates_successfully(self, pipeline):
        """Test that pipeline initializes without errors"""
        assert pipeline is not None
        assert pipeline.config is not None
    
    def test_health_check(self, pipeline):
        """Test health check returns status"""
        health = pipeline.health_check()
        assert isinstance(health, dict)
        assert 'ai_detector' in health
        assert 'plagiarism_detector' in health
        assert 'humanizer' in health


class TestAIDetection:
    """Test AI detection functionality"""
    
    def test_detect_ai_returns_score(self, pipeline, sample_text):
        """Test that AI detection returns a valid score"""
        result = pipeline.detect_ai(sample_text)
        
        assert isinstance(result, dict)
        assert 'score' in result
        assert 0.0 <= result['score'] <= 1.0
        assert 'label' in result
        assert result['label'] in ['human', 'ai', 'unknown']
    
    def test_detect_ai_with_short_text(self, pipeline):
        """Test AI detection with text below minimum length"""
        with pytest.raises(ValueError):
            pipeline.detect_ai("Short")
    
    def test_detect_ai_with_empty_text(self, pipeline):
        """Test AI detection with empty text"""
        with pytest.raises(ValueError):
            pipeline.detect_ai("")
    
    def test_detect_ai_caching(self, pipeline, sample_text):
        """Test that caching works for AI detection"""
        # First call
        result1 = pipeline.detect_ai(sample_text, use_cache=True)
        
        # Second call (should use cache)
        result2 = pipeline.detect_ai(sample_text, use_cache=True)
        
        assert result1['score'] == result2['score']


class TestPlagiarismDetection:
    """Test plagiarism detection functionality"""
    
    def test_check_plagiarism_returns_result(self, pipeline, sample_text):
        """Test that plagiarism check returns valid result"""
        result = pipeline.check_plagiarism(sample_text)
        
        assert isinstance(result, dict)
        assert 'score' in result
        assert 0.0 <= result['score'] <= 1.0
        assert 'matches' in result
        assert 'total_matches' in result
        assert isinstance(result['matches'], list)
    
    def test_check_plagiarism_with_short_text(self, pipeline):
        """Test plagiarism check with text below minimum length"""
        with pytest.raises(ValueError):
            pipeline.check_plagiarism("Short")
    
    def test_check_plagiarism_match_structure(self, pipeline, sample_text):
        """Test that plagiarism matches have correct structure"""
        result = pipeline.check_plagiarism(sample_text)
        
        if result['matches']:
            match = result['matches'][0]
            assert 'source' in match
            assert 'similarity' in match


class TestHumanization:
    """Test humanization functionality"""
    
    def test_humanize_returns_result(self, pipeline, sample_text):
        """Test that humanization returns valid result"""
        result = pipeline.humanize(sample_text)
        
        assert isinstance(result, dict)
        assert 'text' in result
        assert 'ai_score_before' in result
        assert 'ai_score_after' in result
        assert 'iterations' in result
        assert isinstance(result['text'], str)
    
    def test_humanize_with_short_text(self, pipeline):
        """Test humanization with text below minimum length"""
        with pytest.raises(ValueError):
            pipeline.humanize("Short")
    
    def test_humanize_scores_valid(self, pipeline, sample_text):
        """Test that humanization scores are valid"""
        result = pipeline.humanize(sample_text)
        
        assert 0.0 <= result['ai_score_before'] <= 1.0
        assert 0.0 <= result['ai_score_after'] <= 1.0


class TestCompleteAnalysis:
    """Test complete analysis workflow"""
    
    def test_analyze_all_modules(self, pipeline, sample_text):
        """Test running all analyses together"""
        result = pipeline.analyze(
            sample_text,
            check_ai=True,
            check_plagiarism=True,
            humanize=True
        )
        
        assert isinstance(result, dict)
        assert 'ai_detection' in result
        assert 'plagiarism' in result
        assert 'humanization' in result
        assert 'processing_time' in result
    
    def test_analyze_selective_modules(self, pipeline, sample_text):
        """Test running only selected analyses"""
        result = pipeline.analyze(
            sample_text,
            check_ai=True,
            check_plagiarism=False,
            humanize=False
        )
        
        assert 'ai_detection' in result
        assert 'plagiarism' not in result
        assert 'humanization' not in result
    
    def test_analyze_processing_time(self, pipeline, sample_text):
        """Test that processing time is recorded"""
        result = pipeline.analyze(sample_text)
        
        assert 'processing_time' in result
        assert result['processing_time'] > 0


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_very_long_text(self, pipeline):
        """Test with text at maximum length"""
        long_text = "a" * 50000
        result = pipeline.detect_ai(long_text)
        assert result is not None
    
    def test_text_over_limit(self, pipeline):
        """Test with text exceeding maximum length"""
        too_long = "a" * 50001
        with pytest.raises(ValueError):
            pipeline.detect_ai(too_long)
    
    def test_unicode_text(self, pipeline):
        """Test with Unicode characters"""
        unicode_text = "Hello 世界 🌍 Привет مرحبا " * 10
        result = pipeline.detect_ai(unicode_text)
        assert result is not None
    
    def test_special_characters(self, pipeline):
        """Test with special characters"""
        special_text = "Test @#$%^&*() <html> {code} [brackets] " * 10
        result = pipeline.detect_ai(special_text)
        assert result is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
