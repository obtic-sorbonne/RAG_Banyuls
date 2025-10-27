class EntityAugmenter:
    def augment_context(self, context: str, query_entities: dict) -> str:
        """Highlight relevant entities in context"""
        if not query_entities:
            return context
        
        augmented = []
        for item in context.split("\n\n"):
            # Highlight matching entities
            for entity_type, values in query_entities.items():
                if not isinstance(values, list):
                    values = [values]
                
                for value in values:
                    if value.lower() in item.lower():
                        # Add entity annotation
                        item = item.replace(
                            value, 
                            f"[{entity_type.upper()}: {value}]"
                        )
            augmented.append(item)
        
        return "\n\n".join(augmented)
