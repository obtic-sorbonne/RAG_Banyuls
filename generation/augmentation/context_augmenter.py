from typing import List, Dict
from datetime import datetime

from .temporal_augmenter import TemporalAugmenter
from .entity_augmenter import EntityAugmenter

class ContextAugmenter:
    def __init__(self):
        self.temporal_aug = TemporalAugmenter()
        self.entity_aug = EntityAugmenter()

    class ContextAugmenter:
        def __init__(self):
            pass

    def augment_with_metadata(self, results: List[Dict]) -> str:
        """Augment context with metadata information"""
        augmented_context = []

        for result in results:
            metadata = result.get('metadata', {})
            content = result.get('content', '')

            # Create citation string
            citation = f"[Source: Livre {metadata.get('book', 'Inconnu')}, Page {metadata.get('page', 'Inconnue')}]"

            # Add temporal information if available
            temporal_info = ""
            if 'primary_year' in metadata:
                temporal_info = f" (Année principale: {metadata['primary_year']})"

            augmented_context.append(f"{content}{temporal_info}\n{citation}\n")

        return "\n".join(augmented_context)

    def filter_by_temporal_constraints(self, results: List[Dict],
                                       start_date: datetime = None,
                                       end_date: datetime = None) -> List[Dict]:
        """Filter results based on temporal constraints"""
        if not start_date and not end_date:
            return results

        filtered_results = []
        for result in results:
            metadata = result.get('metadata', {})
            years = metadata.get('years', '').split(',')

            # Check if any year falls within range
            for year in years:
                try:
                    year_int = int(year.strip())
                    if ((not start_date or year_int >= start_date.year) and
                            (not end_date or year_int <= end_date.year)):
                        filtered_results.append(result)
                        break
                except ValueError:
                    continue

        return filtered_results
    def augment(self, context: str, query: str, query_metadata: dict) -> str:
        """Apply all augmentation strategies"""
        # Apply temporal augmentation
        context = self.temporal_aug.augment_context(
            context, 
            query_metadata.get("temporal", {}).get("start")
        )
        
        # Apply entity highlighting
        context = self.entity_aug.augment_context(
            context, 
            query_metadata.get("entities", {})
        )
        
        return context
