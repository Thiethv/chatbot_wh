# app/utils/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
query_counter = Counter('chatbot_queries_total', 'Total number of queries')
response_time = Histogram('chatbot_response_time_seconds', 'Response time in seconds')
confidence_gauge = Gauge('chatbot_confidence_score', 'Average confidence score')

class MetricsCollector:
    @staticmethod
    def record_query(duration: float, confidence: float):
        query_counter.inc()
        response_time.observe(duration)
        confidence_gauge.set(confidence)