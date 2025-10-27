from datetime import datetime, timedelta
import re

class TemporalAugmenter:
    def augment_context(self, context: str, query_date: datetime) -> str:
        """Enhance context with temporal relationships"""
        if not query_date:
            return context
        
        # Add temporal context to each context item
        augmented = []
        for item in context.split("\n\n"):
            if "date" in item.lower():
                # Already has date information
                augmented.append(item)
            else:
                # Extract and add inferred date
                item_date = self._infer_date(item)
                if item_date:
                    date_diff = (item_date - query_date).days
                    time_desc = self._time_description(date_diff)
                    augmented.append(f"[Date: {item_date.date()} ({time_desc})]\n{item}")
                else:
                    augmented.append(f"[Date: Unknown]\n{item}")
        
        return "\n\n".join(augmented)
    
    def _infer_date(self, text: str) -> datetime:
        """Infer date from context text"""
        # Try explicit date patterns
        patterns = [
            r"\d{4}-\d{2}-\d{2}",  # YYYY-MM-DD
            r"\d{1,2}/\d{1,2}/\d{4}",  # DD/MM/YYYY
            r"\d{1,2}\s+[A-Za-z]+\s+\d{4}"  # 14 juillet 1832
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return datetime.strptime(match.group(0), "%Y-%m-%d")
                except:
                    pass
                try:
                    return datetime.strptime(match.group(0), "%d/%m/%Y")
                except:
                    pass
                try:
                    # French month handling
                    parts = match.group(0).split()
                    if len(parts) == 3:
                        day = int(parts[0])
                        month = self._month_to_number(parts[1])
                        year = int(parts[2])
                        return datetime(year, month, day)
                except:
                    continue
        
        # Fallback to metadata extraction
        year_match = re.search(r"\b(18\d{2}|19\d{2})\b", text)
        if year_match:
            return datetime(int(year_match.group(0)), 1, 1)
        
        return None
    
    def _month_to_number(self, month_str: str) -> int:
        months = {
            'janvier': 1, 'février': 2, 'mars': 3, 'avril': 4,
            'mai': 5, 'juin': 6, 'juillet': 7, 'août': 8,
            'septembre': 9, 'octobre': 10, 'novembre': 11, 'décembre': 12
        }
        return months.get(month_str.lower(), 1)
    
    def _time_description(self, days_diff: int) -> str:
        """Generate human-readable time description"""
        if days_diff == 0:
            return "same day"
        elif abs(days_diff) < 7:
            return f"{abs(days_diff)} days {'before' if days_diff > 0 else 'after'}"
        elif abs(days_diff) < 30:
            weeks = abs(days_diff) // 7
            return f"{weeks} weeks {'before' if days_diff > 0 else 'after'}"
        else:
            months = abs(days_diff) // 30
            return f"{months} months {'before' if days_diff > 0 else 'after'}"
