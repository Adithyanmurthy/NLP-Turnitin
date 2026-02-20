"""
API tests for FastAPI endpoints
Person 4: Test all API routes
"""

import pytest
from fastapi.testclient import TestClient
from api.app import app

@pytest.fixture(scope="module")
def client():
    """Create test client"""
    return TestClient(app)


class TestGeneralEndpoints:
    """Test general API endpoints"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert 'name' in data
        assert 'version' in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert 'ai_detector' in data
        assert 'plagiarism_detector' in data
        assert 'humanizer' in data


class TestAnalyzeEndpoint:
    """Test the main analyze endpoint"""
    
    def test_analyze_with_valid_text(self, client):
        """Test analyze endpoint with valid text"""
        payload = {
            "text": "This is a test text for analysis. " * 10,
            "check_ai": True,
            "check_plagiarism": True,
            "humanize": False
        }
        
        response = client.post("/analyze", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert 'input_length' in data
        assert 'processing_time' in data
        assert 'ai_detection' in data
        assert 'plagiarism' in data
    
    def test_analyze_with_short_text(self, client):
        """Test analyze endpoint with text too short"""
        payload = {
            "text": "Short",
            "check_ai": True
        }
        
        response = client.post("/analyze", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_analyze_with_empty_text(self, client):
        """Test analyze endpoint with empty text"""
        payload = {
            "text": "",
            "check_ai": True
        }
        
        response = client.post("/analyze", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_analyze_ai_only(self, client):
        """Test analyze with only AI detection"""
        payload = {
            "text": "Test text for AI detection only. " * 10,
            "check_ai": True,
            "check_plagiarism": False,
            "humanize": False
        }
        
        response = client.post("/analyze", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert 'ai_detection' in data
        assert 'plagiarism' not in data or data['plagiarism'] is None
    
    def test_analyze_with_humanization(self, client):
        """Test analyze with humanization enabled"""
        payload = {
            "text": "Test text for humanization. " * 10,
            "check_ai": False,
            "check_plagiarism": False,
            "humanize": True
        }
        
        response = client.post("/analyze", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert 'humanization' in data


class TestDetectAIEndpoint:
    """Test the AI detection endpoint"""
    
    def test_detect_ai_valid_text(self, client):
        """Test AI detection with valid text"""
        response = client.post(
            "/detect-ai",
            params={"text": "This is a test text. " * 10}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'score' in data
        assert 'label' in data
        assert 0.0 <= data['score'] <= 1.0
    
    def test_detect_ai_with_cache(self, client):
        """Test AI detection with caching"""
        text = "Cached test text. " * 10
        
        # First request
        response1 = client.post(
            "/detect-ai",
            params={"text": text, "use_cache": True}
        )
        
        # Second request (should use cache)
        response2 = client.post(
            "/detect-ai",
            params={"text": text, "use_cache": True}
        )
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response1.json()['score'] == response2.json()['score']


class TestPlagiarismEndpoint:
    """Test the plagiarism check endpoint"""
    
    def test_check_plagiarism_valid_text(self, client):
        """Test plagiarism check with valid text"""
        response = client.post(
            "/check-plagiarism",
            params={"text": "This is a test text for plagiarism check. " * 10}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'score' in data
        assert 'matches' in data
        assert 'total_matches' in data
        assert 0.0 <= data['score'] <= 1.0


class TestHumanizeEndpoint:
    """Test the humanization endpoint"""
    
    def test_humanize_valid_text(self, client):
        """Test humanization with valid text"""
        response = client.post(
            "/humanize",
            params={"text": "This is AI-generated text to humanize. " * 10}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'text' in data
        assert 'ai_score_before' in data
        assert 'ai_score_after' in data
        assert 'iterations' in data


class TestErrorHandling:
    """Test error handling"""
    
    def test_invalid_json(self, client):
        """Test with invalid JSON"""
        response = client.post(
            "/analyze",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_required_field(self, client):
        """Test with missing required field"""
        payload = {
            "check_ai": True
            # Missing 'text' field
        }
        
        response = client.post("/analyze", json=payload)
        assert response.status_code == 422


class TestRateLimiting:
    """Test rate limiting (if implemented)"""
    
    def test_rate_limit_not_exceeded_normal_use(self, client):
        """Test that normal use doesn't hit rate limit"""
        text = "Test text. " * 10
        
        # Make a few requests
        for _ in range(3):
            response = client.post(
                "/detect-ai",
                params={"text": text}
            )
            assert response.status_code == 200


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
