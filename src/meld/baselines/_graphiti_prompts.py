"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modifications: excerpted from five upstream files (below). The excerpted function bodies
are byte-identical to upstream, except that _clean_input and _strip_code_fences (client methods upstream) are
dedented to module level; pydantic's Message is replaced by a NamedTuple with the same two
fields, and EdgeDuplicate by its JSON schema as pydantic 2 emits it.

Graphiti edge-resolution prompt and helpers, vendored verbatim from graphiti-core 0.30.2.

Source: PyPI sdist graphiti_core-0.30.2.tar.gz
  https://files.pythonhosted.org/packages/4a/a1/3e0fe119b6f3116ef7671ae1b2d98ef70655ac634d9914e748b5f97ecacb/graphiti_core-0.30.2.tar.gz
  sdist sha256 b8ac6999705d350ebb53d799505d49b76f5d74c20265a09df4094645b5d8ada9
  graphiti_core/prompts/dedupe_edges.py sha256 607cb830bb02a6020fdfea013e321345c0c0d5629f7895dc76622130cefc22c6
    lines 24-32 EdgeDuplicate (as schema), lines 43-101 resolve_edge
  graphiti_core/llm_client/client.py sha256 eed64f7c8dc7b9f0e3e72973cfd08b5447b9007c684092d5df402919285304e9
    lines 38-57 get_extraction_language_instruction, lines 98-119 _clean_input
  graphiti_core/llm_client/openai_generic_client.py sha256 62d198e838030f4e309df2eeb0eb7cf1770e3bb32cbe34c8a9f3ee3dc0ca7705
    lines 127-138 _strip_code_fences (a staticmethod upstream, dedented)
  graphiti_core/utils/maintenance/dedup_helpers.py lines 39-42 _normalize_string_exact
The calling logic (graphiti_core/utils/maintenance/edge_operations.py sha256
b773ff4489968af2a996d5074e679cab9806cc0904a7ff9f2aecc74382325abe, resolve_extracted_edge, and
OpenAIGenericClient.generate_response) is re-implemented in meld.baselines.graphiti_policy;
tests/test_baselines.py pins the rendered prompt hash computed from the upstream files.
"""
import re
from typing import Any, NamedTuple


class Message(NamedTuple):
    role: str
    content: str


# EdgeDuplicate.model_json_schema() under pydantic 2 (sort_keys order)
EDGE_DUPLICATE_SCHEMA = {
    "properties": {
        "contradicted_facts": {
            "description": "List of idx values of contradicted facts (from full idx range). Empty list if none.",
            "items": {"type": "integer"}, "title": "Contradicted Facts", "type": "array"},
        "duplicate_facts": {
            "description": "List of idx values of duplicate facts (only from EXISTING FACTS range). Empty list if none.",
            "items": {"type": "integer"}, "title": "Duplicate Facts", "type": "array"},
    },
    "required": ["duplicate_facts", "contradicted_facts"], "title": "EdgeDuplicate", "type": "object",
}


def resolve_edge(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a fact deduplication assistant. '
            'NEVER mark facts with key differences as duplicates.',
        ),
        Message(
            role='user',
            content=f"""
NEVER mark facts as duplicates if they have key differences, particularly around numeric values, dates, or key qualifiers.

IMPORTANT constraints:
- duplicate_facts: ONLY idx values from EXISTING FACTS (NEVER include FACT INVALIDATION CANDIDATES)
- contradicted_facts: idx values from EITHER list (EXISTING FACTS or FACT INVALIDATION CANDIDATES)
- The idx values are continuous across both lists (INVALIDATION CANDIDATES start where EXISTING FACTS end)

<EXISTING FACTS>
{context['existing_edges']}
</EXISTING FACTS>

<FACT INVALIDATION CANDIDATES>
{context['edge_invalidation_candidates']}
</FACT INVALIDATION CANDIDATES>

<NEW FACT>
{context['new_edge']}
</NEW FACT>

You will receive TWO lists of facts with CONTINUOUS idx numbering across both lists.
EXISTING FACTS are indexed first, followed by FACT INVALIDATION CANDIDATES.

1. DUPLICATE DETECTION:
   - If the NEW FACT represents identical factual information as any fact in EXISTING FACTS, return those idx values in duplicate_facts.
   - If no duplicates, return an empty list for duplicate_facts.

2. CONTRADICTION DETECTION:
   - Determine which facts the NEW FACT contradicts from either list.
   - A fact from EXISTING FACTS can be both a duplicate AND contradicted (e.g., semantically the same but the new fact updates/supersedes it).
   - Return all contradicted idx values in contradicted_facts.
   - If no contradictions, return an empty list for contradicted_facts.

<EXAMPLE>
EXISTING FACT: idx=0, "Alice joined Acme Corp in 2020"
NEW FACT: "Alice joined Acme Corp in 2020"
Result: duplicate_facts=[0], contradicted_facts=[] (identical factual information)

EXISTING FACT: idx=1, "Alice works at Acme Corp as a software engineer"
NEW FACT: "Alice works at Acme Corp as a senior engineer"
Result: duplicate_facts=[], contradicted_facts=[1] (same relationship but updated title — contradiction, NOT a duplicate)

EXISTING FACT: idx=2, "Bob ran 5 miles on Tuesday"
NEW FACT: "Bob ran 3 miles on Wednesday"
Result: duplicate_facts=[], contradicted_facts=[] (different events on different days — neither duplicate nor contradiction)
</EXAMPLE>
""",
        ),
    ]



def get_extraction_language_instruction(group_id: str | None = None) -> str:
    """Returns instruction for language extraction behavior.

    Override this function to customize language extraction:
    - Return empty string to disable multilingual instructions
    - Return custom instructions for specific language requirements
    - Use group_id to provide different instructions per group/partition

    Args:
        group_id: Optional partition identifier for the graph

    Returns:
        str: Language instruction to append to system messages
    """
    return (
        '\n\nAny extracted information should be returned in the same language as it was written in. '
        'Only output non-English text when the user has written full sentences or phrases in that non-English language. '
        'Otherwise, output English.'
    )



def _clean_input(self, input: str) -> str:
    """Clean input string of invalid unicode and control characters.

    Args:
        input: Raw input string to be cleaned

    Returns:
        Cleaned string safe for LLM processing
    """
    # Clean any invalid Unicode
    cleaned = input.encode('utf-8', errors='ignore').decode('utf-8')

    # Remove zero-width characters and other invisible unicode
    zero_width = '\u200b\u200c\u200d\ufeff\u2060'
    for char in zero_width:
        cleaned = cleaned.replace(char, '')

    # Remove control characters except newlines, returns, and tabs
    cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\n\r\t')

    return cleaned



def _strip_code_fences(text: str) -> str:
    """Strip a wrapping markdown code fence from a JSON payload.

    OpenAI-compatible models served via Ollama/llama.cpp etc. frequently wrap their
    output in a ```json … ``` fence even when a json_schema/json_object response_format
    is requested, which breaks a bare ``json.loads``. No-op when there is no fence.
    """
    stripped = text.strip()
    if stripped.startswith('```'):
        stripped = re.sub(r'^```[a-zA-Z0-9_-]*[ \t]*\r?\n?', '', stripped)
        stripped = re.sub(r'\r?\n?```[ \t]*$', '', stripped)
    return stripped.strip()


def _normalize_string_exact(name: str) -> str:
    """Lowercase text and collapse whitespace so equal names map to the same key."""
    normalized = re.sub(r'[\s]+', ' ', name.lower())
    return normalized.strip()
