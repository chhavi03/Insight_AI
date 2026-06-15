from typing import Dict, Any, List

class SignalFusionEngine:
    """
    Mathematical fusion engine that synthesizes independent telemetry data streams
    (Audio metrics, NLP text insights, and Vision array scores) into a singular composite score.
    """
    def __init__(self, clarity_weight: float = 0.40, pace_weight: float = 0.30, visual_weight: float = 0.30):
        # Ensure weights normalize perfectly to 1.0 (100%)
        total_weight = clarity_weight + pace_weight + visual_weight
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError("Mathematical bounds violation: Target fusion weights must sum precisely to 1.0")
            
        self.w_clarity = clarity_weight
        self.w_pace = pace_weight
        self.w_visual = visual_weight
        print(f"[SIGNAL FUSION] Matrix loaded. Ratios -> Clarity: {self.w_clarity}, Pace: {self.w_pace}, Visual: {self.w_visual}")

    def generate_composite_profile(self, text_analysis: Dict[str, Any], visual_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Ingests the text analytic blocks and a historical array of frame snapshots, 
        applies matrix weighting, and compiles a comprehensive performance scorecard.
        """
        # 1. Extract raw textual metrics
        clarity_score = float(text_analysis.get("clarity_score", 1.0)) # Range: 0.0 - 1.0
        pace_score = float(text_analysis.get("pace", {}).get("score", 1.0))  # Range: 0.0 - 1.0
        
        # 2. Process visual history array to find average face engagement and confidence
        total_frames = len(visual_history)
        if total_frames == 0:
            visual_score = 1.0  # Safe neutral default if no frames were collected
            dominant_expression = "neutral"
        else:
            detected_faces = [f for f in visual_history if f.get("face_detected", False)]
            face_ratio = len(detected_faces) / total_frames
            
            # Map dominant emotion trends across the history slice
            emotion_counts = {}
            for frame in detected_faces:
                emotion = frame.get("dominant_emotion", "neutral")
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                
            dominant_expression = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral"
            
            # Professional composure score: penalty applied if candidate is completely out of frame
            visual_score = face_ratio 

        # 3. Calculate Weighted Composite Formula
        # Formula: S_composite = (S_clarity * W_clarity) + (S_pace * W_pace) + (S_visual * W_visual)
        composite_raw = (clarity_score * self.w_clarity) + (pace_score * self.w_pace) + (visual_score * self.w_visual)
        composite_final_percentage = round(composite_raw * 100.0, 1)

        return {
            "overall_performance_score": composite_final_percentage,
            "metrics_breakdown": {
                "clarity_rating": round(clarity_score, 2),
                "pace_rating": round(pace_score, 2),
                "visual_presence_rating": round(visual_score, 2)
            },
            "dominant_visual_trait": dominant_expression,
            "data_completeness": {
                "analyzed_frames": total_frames,
                "words_processed": len(text_analysis.get("sentiment", {})) # Structural proxy
            }
        }

if __name__ == "__main__":
    # Test integration matrix simulation
    fusion = SignalFusionEngine()
    
    # Mock data payloads simulating outputs from Step 7 and Step 8
    mock_text = {
        "clarity_score": 0.85, # Good speech clarity
        "pace": {"score": 0.90, "wpm": 142, "status": "ideal"} # Perfect pace
    }
    
    mock_visual_stream = [
        {"face_detected": True, "dominant_emotion": "happy"},
        {"face_detected": True, "dominant_emotion": "neutral"},
        {"face_detected": False, "dominant_emotion": "neutral"}, # Simulated head turn / occlusion
        {"face_detected": True, "dominant_emotion": "happy"}
    ]
    
    fused_results = fusion.generate_composite_profile(mock_text, mock_visual_stream)
    print("\n[FUSION ENGINE OUTPUT]:")
    print(fused_results)