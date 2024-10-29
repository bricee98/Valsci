from typing import List
from app.models.claim import Claim

class BatchJob:
    def __init__(self, claims: List[Claim], status: str = 'pending'):
        self.claims = claims
        self.status = status

    def __repr__(self):
        return f"<BatchJob(status={self.status}, claims_count={len(self.claims)})>"
