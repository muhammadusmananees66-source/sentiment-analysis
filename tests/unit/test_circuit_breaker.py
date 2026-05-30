"""Tests for circuit breaker pattern"""
import pytest
import time
from src.serving.circuit_breaker import CircuitBreaker


class TestCircuitBreaker:
    """Test cases for CircuitBreaker"""

    def test_initial_state_closed(self):
        """Test initial state is CLOSED"""
        cb = CircuitBreaker()
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0

    def test_record_failure(self):
        """Test recording failures"""
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        assert cb.failure_count == 1
        assert cb.state == "CLOSED"

        cb.record_failure()
        cb.record_failure()
        assert cb.state == "OPEN"

    def test_circuit_opens_after_threshold(self):
        """Test circuit opens after threshold exceeded"""
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        assert cb.is_open() is False
        cb.record_failure()
        assert cb.is_open() is True

    def test_circuit_half_open_after_timeout(self):
        """Test circuit becomes half-open after timeout"""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        cb.record_failure()
        assert cb.is_open() is True

        time.sleep(0.2)
        assert cb.is_open() is False
        assert cb.state == "HALF_OPEN"

    def test_record_success_resets_circuit(self):
        """Test success resets circuit from half-open"""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1)
        cb.record_failure()
        
        # Force half-open state for testing
        cb.state = "HALF_OPEN"
        
        cb.record_success()
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0