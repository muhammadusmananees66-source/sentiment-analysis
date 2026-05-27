# src/serving/circuit_breaker.py
import time

class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascade failures
    States: CLOSED (working) -> OPEN (failed) -> HALF-OPEN (testing)
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout  # seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def is_open(self) -> bool:
        """Check if circuit is open (should reject requests)"""
        if self.state == "OPEN":
            # Check if recovery timeout has elapsed
            if self.last_failure_time and time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return False
            return True
        return False
    
    def record_failure(self):
        """Record a failure and potentially open circuit"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            print(f"Circuit breaker OPEN after {self.failure_count} failures")
            
    def record_success(self):
        """Record a success and reset circuit"""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0
            print("Circuit breaker CLOSED (recovered)")
        elif self.state == "CLOSED":
            self.failure_count = 0