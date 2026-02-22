"""
backend/session.py — In-memory session state for NDA Guardian API.

Tracks loaded document clauses and per-session query statistics.
"""

from dataclasses import dataclass, field


@dataclass
class Session:
    # Loaded document
    clauses: dict[str, str] = field(default_factory=dict)
    document_loaded: bool = False

    # Query statistics
    query_count: int = 0
    local_count: int = 0
    cloud_count: int = 0
    total_latency_ms: float = 0.0
    total_words_sent_to_cloud: int = 0
    total_cost_usd: float = 0.0

    def record_query(
        self,
        source: str,
        latency_ms: float,
        words_sent: int = 0,
    ) -> None:
        """Update stats after a query completes."""
        self.query_count += 1
        self.total_latency_ms += latency_ms

        if source == "on-device":
            self.local_count += 1
        else:
            self.cloud_count += 1
            self.total_words_sent_to_cloud += words_sent
            # Approximate Gemini 2.0 Flash cost: ~$0.00001 per word sent
            self.total_cost_usd += words_sent * 0.00001

    def stats(self) -> dict:
        """Return a snapshot of session statistics."""
        local_pct = (
            round(100 * self.local_count / self.query_count)
            if self.query_count > 0 else 0
        )
        cloud_pct = (
            round(100 * self.cloud_count / self.query_count)
            if self.query_count > 0 else 0
        )
        avg_latency = (
            round(self.total_latency_ms / self.query_count, 1)
            if self.query_count > 0 else 0.0
        )
        return {
            "query_count": self.query_count,
            "local_count": self.local_count,
            "cloud_count": self.cloud_count,
            "local_pct": local_pct,
            "cloud_pct": cloud_pct,
            "avg_latency_ms": avg_latency,
            "total_words_sent_to_cloud": self.total_words_sent_to_cloud,
            "total_cost_usd": round(self.total_cost_usd, 6),
        }

    def reset(self) -> None:
        """Reset all session state."""
        self.clauses = {}
        self.document_loaded = False
        self.query_count = 0
        self.local_count = 0
        self.cloud_count = 0
        self.total_latency_ms = 0.0
        self.total_words_sent_to_cloud = 0
        self.total_cost_usd = 0.0


# Singleton session instance
session = Session()
