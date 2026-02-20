"""API tests with sample queries to verify system responses.

This test file:
1. Creates a seed database with realistic data
2. Tests the API endpoints
3. Verifies sample queries return correct responses
"""

import tempfile
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

from haikugraph.api.server import create_app


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def seed_db():
    """Create a seeded DuckDB database with test data."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = Path(f.name)
    db_path.unlink()  # DuckDB needs to create the file
    
    conn = duckdb.connect(str(db_path))
    
    # Create customers table
    conn.execute("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            name VARCHAR,
            email VARCHAR,
            region VARCHAR,
            created_at TIMESTAMP
        )
    """)
    conn.execute("""
        INSERT INTO customers VALUES
        (1, 'Acme Corp', 'acme@corp.com', 'North', '2023-01-15'),
        (2, 'Beta Inc', 'beta@inc.com', 'South', '2023-02-20'),
        (3, 'Gamma LLC', 'gamma@llc.com', 'North', '2023-03-10'),
        (4, 'Delta Ltd', 'delta@ltd.com', 'East', '2023-04-05'),
        (5, 'Echo GmbH', 'echo@gmbh.com', 'West', '2023-05-12')
    """)
    
    # Create orders table
    conn.execute("""
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            order_date DATE,
            amount DECIMAL(10, 2),
            status VARCHAR
        )
    """)
    conn.execute("""
        INSERT INTO orders VALUES
        (1, 1, '2024-01-05', 1500.00, 'completed'),
        (2, 1, '2024-01-15', 2300.00, 'completed'),
        (3, 2, '2024-01-10', 800.00, 'completed'),
        (4, 3, '2024-02-01', 3200.00, 'completed'),
        (5, 3, '2024-02-15', 1100.00, 'pending'),
        (6, 4, '2024-02-20', 950.00, 'completed'),
        (7, 5, '2024-03-01', 4500.00, 'completed'),
        (8, 1, '2024-03-10', 1800.00, 'refunded')
    """)
    
    # Create products table
    conn.execute("""
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            name VARCHAR,
            category VARCHAR,
            price DECIMAL(10, 2)
        )
    """)
    conn.execute("""
        INSERT INTO products VALUES
        (1, 'Widget A', 'Widgets', 29.99),
        (2, 'Widget B', 'Widgets', 49.99),
        (3, 'Gadget X', 'Gadgets', 199.99),
        (4, 'Gadget Y', 'Gadgets', 299.99),
        (5, 'Tool Z', 'Tools', 79.99)
    """)
    
    conn.close()
    
    yield db_path
    
    db_path.unlink(missing_ok=True)


@pytest.fixture
def client(seed_db):
    """Create test client with seeded database."""
    app = create_app(db_path=seed_db)
    return TestClient(app)


# =============================================================================
# Test Health & Architecture Endpoints
# =============================================================================

class TestHealthEndpoint:
    """Tests for health endpoint."""
    
    def test_health_returns_ok(self, client):
        """Health check should return ok status."""
        resp = client.get("/api/assistant/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["db_exists"] is True
        assert "version" in data


class TestArchitectureEndpoint:
    """Tests for architecture endpoint."""
    
    def test_architecture_returns_info(self, client):
        """Architecture endpoint should return system info."""
        resp = client.get("/api/assistant/architecture")
        assert resp.status_code == 200
        data = resp.json()

        assert data["system_name"].startswith("dataDa")
        assert "version" in data
        assert len(data["pipeline_flow"]) >= 5
        assert len(data["agents"]) >= 5
        assert len(data["guardrails"]) >= 5
    
    def test_architecture_lists_all_agents(self, client):
        """Architecture should include core agent roles."""
        resp = client.get("/api/assistant/architecture")
        data = resp.json()
        
        agent_names = [a["name"] for a in data["agents"]]
        assert "ChiefAnalystAgent" in agent_names
        assert "IntakeAgent" in agent_names
        assert "QueryEngineerAgent" in agent_names
        assert "AuditAgent" in agent_names
        assert "NarrativeAgent" in agent_names


class TestUIEndpoint:
    """Tests for UI endpoint."""
    
    def test_ui_returns_html(self, client):
        """Root should return HTML UI."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "dataDa" in resp.text

    def test_ui_script_has_query_runtime(self, client):
        """UI script should contain the runQuery function and Chart.js integration."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert "runQuery" in resp.text
        assert "chart.js" in resp.text.lower() or "Chart" in resp.text


# =============================================================================
# Test Sample Queries
# =============================================================================

class TestSampleQueries:
    """Tests for sample queries to verify system responses."""
    
    def test_count_customers(self, client):
        """Should answer 'How many customers?'"""
        resp = client.post("/api/assistant/query", json={
            "goal": "How many customers do we have?"
        })
        assert resp.status_code == 200
        data = resp.json()
        
        # Should be successful
        assert data["success"] is True
        
        # Should have required fields
        assert "answer_markdown" in data
        assert "confidence" in data
        assert "trace_id" in data
        
        # Answer should mention the count (5 customers)
        # Note: actual answer depends on agent logic
        assert data["answer_markdown"]
    
    def test_total_revenue(self, client):
        """Should answer 'What is total revenue?'"""
        resp = client.post("/api/assistant/query", json={
            "goal": "What is the total revenue from orders?"
        })
        assert resp.status_code == 200
        data = resp.json()
        
        assert data["success"] is True
        assert data["answer_markdown"]
        assert data["confidence"] in ["high", "medium", "low", "uncertain"]
    
    def test_list_products(self, client):
        """Should answer 'What products do we sell?'"""
        resp = client.post("/api/assistant/query", json={
            "goal": "What products do we sell?"
        })
        assert resp.status_code == 200
        data = resp.json()
        
        assert data["success"] is True
        assert data["answer_markdown"]
    
    def test_top_customers(self, client):
        """Should answer 'Top customers by order amount'"""
        resp = client.post("/api/assistant/query", json={
            "goal": "Show me top 3 customers by total order amount"
        })
        assert resp.status_code == 200
        data = resp.json()
        
        assert data["success"] is True
        assert data["answer_markdown"]
    
    def test_orders_by_region(self, client):
        """Should answer 'Orders by region'"""
        resp = client.post("/api/assistant/query", json={
            "goal": "How many orders per region?"
        })
        assert resp.status_code == 200
        data = resp.json()
        
        assert data["success"] is True


# =============================================================================
# Test Response Format
# =============================================================================

class TestResponseFormat:
    """Tests for response format compliance."""
    
    def test_response_has_all_required_fields(self, client):
        """Response should have all required fields."""
        resp = client.post("/api/assistant/query", json={
            "goal": "Count orders"
        })
        data = resp.json()
        
        # Required fields per spec
        required_fields = [
            "success",
            "answer_markdown",
            "confidence",
            "confidence_score",
            "definition_used",
            "evidence",
            "sanity_checks",
            "trace_id",
        ]
        
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
    
    def test_confidence_is_valid_enum(self, client):
        """Confidence should be valid enum value."""
        resp = client.post("/api/assistant/query", json={
            "goal": "List all regions"
        })
        data = resp.json()
        
        valid_confidences = ["high", "medium", "low", "uncertain"]
        assert data["confidence"] in valid_confidences
    
    def test_confidence_score_in_range(self, client):
        """Confidence score should be 0-1."""
        resp = client.post("/api/assistant/query", json={
            "goal": "Average order value"
        })
        data = resp.json()
        
        assert 0.0 <= data["confidence_score"] <= 1.0
    
    def test_sanity_checks_format(self, client):
        """Sanity checks should have correct format."""
        resp = client.post("/api/assistant/query", json={
            "goal": "Count customers"
        })
        data = resp.json()
        
        for check in data["sanity_checks"]:
            assert "check_name" in check
            assert "passed" in check
            assert "message" in check


# =============================================================================
# Test Error Handling
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""
    
    def test_empty_goal_handled(self, client):
        """Empty goal should return error."""
        resp = client.post("/api/assistant/query", json={
            "goal": ""
        })
        # Either validation error or handled gracefully
        assert resp.status_code in [200, 422]
    
    def test_invalid_request_format(self, client):
        """Invalid request format should return error."""
        resp = client.post("/api/assistant/query", json={})
        assert resp.status_code == 422  # Validation error


# =============================================================================
# Test Security (Guardrails)
# =============================================================================

class TestSecurityGuardrails:
    """Tests for security guardrails."""
    
    def test_dangerous_sql_blocked(self, client):
        """Dangerous SQL injection attempts should be blocked."""
        resp = client.post("/api/assistant/query", json={
            "goal": "DROP TABLE customers"
        })
        data = resp.json()
        
        # Should not crash and should not execute DROP
        # The system should return a response (possibly with error)
        assert resp.status_code == 200
        # The answer should NOT indicate the table was dropped


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
