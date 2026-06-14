import re
from typing import Dict, Any
from transformers import pipeline

class TextAnalysisEngine:
    """
    Local NLP Engine that runs sentiment scoring, tracks filler word usage (clarity),
    and evaluates delivery speed (pace/WPM) against professional speech standards.
    """
    def __init__(self):
        print("[TEXT ANALYTICS] Initializing DistilBERT sentiment pipeline...")
        # Load the specified smaller, highly optimized model for fast CPU inference
        self.sentiment_pipeline = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        # Professional standard verbal filler tracking list
        self.filler_words = ["um", "uh", "like", "you know", "basically"]
        print("[TEXT ANALYTICS] Engine fully loaded and ready.")

    def calculate_clarity_score(self, text: str) -> float:
        """
        Calculates the ratio of filler words to total words.
        Returns a score between 0.0 (poor) and 1.0 (perfectly clear).
        Formula: 1.0 - (total_fillers / total_words)
        """
        words = text.lower().split()
        total_words = len(words)
        if total_words == 0:
            return 1.0

        total_fillers = 0
        for filler in self.filler_words:
            # \b ensures we match word boundaries (e.g., 'like' but not 'likely')
            pattern = rf"\b{re.escape(filler)}\b"
            matches = re.findall(pattern, text.lower())
            total_fillers += len(matches)

        filler_ratio = total_fillers / total_words
        # Clamp score between 0.0 and 1.0
        clarity_score = max(0.0, 1.0 - filler_ratio)
        return round(clarity_score, 2)

    def calculate_pace_score(self, text: str, duration_seconds: float) -> Dict[str, Any]:
        """
        Calculates words per minute (WPM) and scores performance against
        the professional benchmark range of 130-160 WPM.
        """
        words = text.split()
        total_words = len(words)
        
        if duration_seconds <= 0 or total_words == 0:
            return {"wpm": 0, "score": 1.0, "status": "ideal"}

        wpm = (total_words / duration_seconds) * 60.0
        wpm_rounded = int(round(wpm))

        # Evaluate score based on divergence from standard conversational ranges
        if 130 <= wpm_rounded <= 160:
            score = 1.0
            status = "ideal"
        elif wpm_rounded < 130:
            # Too slow: scale down down to a floor of 0.0
            score = max(0.0, wpm_rounded / 130.0)
            status = "too slow"
        else:
            # Too fast: drop score as speed exceeds 160
            score = max(0.0, 1.0 - ((wpm_rounded - 160) / 100.0))
            status = "too fast"

        return {
            "wpm": wpm_rounded,
            "score": round(score, 2),
            "status": status
        }

    def analyze_transcript(self, text: str, duration_seconds: float) -> Dict[str, Any]:
        """
        Executes a complete multi-modal text pipeline analysis pass on a transcript block.
        """
        if not text.strip():
            return {
                "sentiment": {"label": "NEUTRAL", "score": 1.0},
                "clarity_score": 1.0,
                "pace": {"wpm": 0, "score": 1.0, "status": "ideal"}
            }

        # 1. Run local Transformer inference
        sentiment_result = self.sentiment_pipeline(text)[0]
        
        # 2. Run algorithmic scores
        clarity = self.calculate_clarity_score(text)
        pace = self.calculate_pace_score(text, duration_seconds)

        return {
            "sentiment": {
                "label": sentiment_result["label"],
                "score": round(sentiment_result["score"], 2)
            },
            "clarity_score": clarity,
            "pace": pace
        }

if __name__ == "__main__":
    # Self-contained validation harness
    engine = TextAnalysisEngine()
    sample_text = "Um, basic speaking is, like, okay, but you know, we need to be clear."
    results = engine.analyze_transcript(sample_text, duration_seconds=15.0)
    print("\n[TEST RESULTS]:")
    print(results)