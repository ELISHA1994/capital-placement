"""
Dictionary Repository

Repository for static skill/job title dictionaries with in-memory Trie for fast lookups.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict
import structlog

from app.domain.interfaces import ISuggestionSourceRepository
from app.domain.value_objects import SearchSuggestion, SuggestionSource, TenantId

logger = structlog.get_logger(__name__)


class DictionaryRepository(ISuggestionSourceRepository):
    """
    Repository for static skill/job title dictionaries.

    Loads dictionaries from JSON files and provides fast prefix matching
    using in-memory trie structure for optimal performance (<1ms lookups).
    """

    def __init__(self, dictionary_path: Path):
        self.dictionary_path = dictionary_path
        self._terms: Dict[str, List[Dict]] = {}  # category -> terms
        self._trie: Dict = {}  # Trie for fast prefix matching
        self._initialized = False

    async def initialize(self) -> None:
        """Load and index all dictionary files."""
        if self._initialized:
            return

        try:
            # Load all dictionary files
            for dict_file in self.dictionary_path.glob("*.json"):
                category = dict_file.stem

                with open(dict_file, 'r') as f:
                    data = json.load(f)

                self._terms[category] = []

                # Extract and index terms
                if 'categories' in data:
                    # Skills.json format with nested categories
                    for cat_name, cat_data in data['categories'].items():
                        terms = cat_data.get('terms', [])
                        self._terms[category].extend([
                            {
                                'text': term,
                                'category': cat_name,
                                'source': category
                            }
                            for term in terms
                        ])
                elif 'terms' in data:
                    # Job titles format with flat list
                    self._terms[category] = [
                        {
                            'text': term,
                            'category': category,
                            'source': category
                        }
                        for term in data['terms']
                    ]

            # Build trie for fast prefix matching
            self._build_trie()

            self._initialized = True

            total_terms = sum(len(terms) for terms in self._terms.values())
            logger.info(
                "Dictionary repository initialized",
                categories=len(self._terms),
                total_terms=total_terms
            )

        except Exception as e:
            logger.error("Failed to initialize dictionary repository", error=str(e))
            raise

    def _build_trie(self) -> None:
        """
        Build trie structure for O(k) prefix lookups where k = prefix length.

        Trie structure allows extremely fast prefix matching without scanning
        all terms. Each character in a term becomes a node in the tree.
        """
        self._trie = {}

        for category, terms in self._terms.items():
            for term_data in terms:
                term = term_data['text'].lower()

                # Insert into trie
                node = self._trie
                for char in term:
                    if char not in node:
                        node[char] = {}
                    node = node[char]

                # Store term data at leaf
                if '__terms__' not in node:
                    node['__terms__'] = []
                node['__terms__'].append(term_data)

    async def get_suggestions(
        self,
        prefix: str,
        tenant_id: TenantId,
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[SearchSuggestion]:
        """
        Get dictionary terms matching prefix using Trie lookup.

        Args:
            prefix: Query prefix to match
            tenant_id: Tenant context (not used for static dictionaries)
            user_id: User context (not used for static dictionaries)
            limit: Maximum suggestions to return

        Returns:
            List of SearchSuggestion objects from dictionaries
        """
        if not self._initialized:
            await self.initialize()

        try:
            prefix_lower = prefix.lower()

            # Traverse trie to find prefix node - O(k) where k = prefix length
            node = self._trie
            for char in prefix_lower:
                if char not in node:
                    return []  # Prefix not found
                node = node[char]

            # Collect all terms under this prefix
            matching_terms = self._collect_terms(node)

            # Convert to suggestions with scoring
            suggestions = []
            for term_data in matching_terms[:limit]:
                # Base score for dictionary terms
                score = 0.8

                # Boost exact matches
                if term_data['text'].lower() == prefix_lower:
                    score = 0.95

                # Determine source type
                source = (
                    SuggestionSource.SKILL_DICTIONARY
                    if term_data['source'] == 'skills'
                    else SuggestionSource.JOB_TITLE
                )

                suggestions.append(SearchSuggestion(
                    text=term_data['text'],
                    source=source,
                    score=score,
                    frequency=0,  # Not tracked for static terms
                    metadata={
                        'category': term_data['category'],
                        'source': term_data['source']
                    }
                ))

            logger.debug(
                "Dictionary suggestions retrieved",
                prefix=prefix,
                count=len(suggestions)
            )

            return suggestions

        except Exception as e:
            logger.error(
                "Failed to get dictionary suggestions",
                error=str(e),
                prefix=prefix
            )
            # Return empty list on error (graceful degradation)
            return []

    def _collect_terms(self, node: Dict, collected: Optional[List] = None) -> List:
        """
        Recursively collect all terms from trie node.

        Args:
            node: Current trie node
            collected: Accumulated terms list

        Returns:
            List of all term data under this node
        """
        if collected is None:
            collected = []

        # Add terms at current node
        if '__terms__' in node:
            collected.extend(node['__terms__'])

        # Recurse into children (skip __terms__ key)
        for key, child_node in node.items():
            if key != '__terms__':
                self._collect_terms(child_node, collected)

        return collected