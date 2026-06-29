import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Install a complete stub for the `openai` SDK before any test module imports the
# app. Individual test files historically did
# `sys.modules.setdefault("openai", SimpleNamespace(OpenAI=object))`; whichever
# module was collected first won the setdefault, and the variants missing
# AsyncOpenAI left `openai` half-stubbed. A test that constructs a provider (which
# calls `openai.AsyncOpenAI(...)`) then failed or passed depending on collection
# order. Stubbing here first makes every test see one consistent, complete stub.
class _StubOpenAIClient:  # tolerates the kwargs providers pass; does nothing
    def __init__(self, *args, **kwargs):
        pass


sys.modules.setdefault(
    "openai",
    types.SimpleNamespace(OpenAI=_StubOpenAIClient, AsyncOpenAI=_StubOpenAIClient),
)
