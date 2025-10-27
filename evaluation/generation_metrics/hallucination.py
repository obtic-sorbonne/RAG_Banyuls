class HallucinationDetector:
    def __init__(self):
        self.entity_matcher = EntityMatcher()
    
    def calculate(self, response: str, context: str) -> dict:
        """Detect and quantify hallucinations in response"""
        # Extract entities from response
        response_entities = self.entity_matcher.extract_entities(response)
        
        # Check against context
        hallucinated = []
        for entity, etype in response_entities.items():
            if entity not in context:
                hallucinated.append({
                    "entity": entity,
                    "type": etype,
                    "context_presence": False
                })
        
        # Calculate metrics
        hallucination_rate = len(hallucinated) / len(response_entities) if response_entities else 0
        
        return {
            "hallucination_rate": hallucination_rate,
            "hallucinated_entities": hallucinated,
            "total_entities": len(response_entities)
        }

class EntityMatcher:
    def extract_entities(self, text: str) -> dict:
        """Extract entities from text (simplified version)"""
        # In production, use spaCy or similar
        entities = {}
        # Match dates
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", text)
        for date in dates:
            entities[date] = "DATE"
        
        # Match measurements
        measurements = re.findall(r"\d+\.?\d*\s*°?[CF]|\d+\s*knots", text)
        for meas in measurements:
            entities[meas] = "MEASUREMENT"
        
        # Match vessel names
        vessels = re.findall(r"[Nn]avire\s+(\w+)", text)
        for vessel in vessels:
            entities[vessel] = "VESSEL"
        
        return entities
