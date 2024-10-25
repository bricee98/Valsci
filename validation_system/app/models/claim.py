from typing import Optional, Dict, Any

class Claim:
    def __init__(self, text: str, source: str = 'user'):
        self.text = text
        self.source = source
        self.status: str = 'queued'
        self.report: Optional[Dict[str, Any]] = None


