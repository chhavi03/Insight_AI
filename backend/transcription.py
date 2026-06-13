import os
from typing import Generator
from faster_whisper import WhisperModel

class TranscriptionEngine:
    """
    High-performance, local machine learning engine managing Whisper audio analytics.
    Utilizes optimized 8-bit quantization (int8) to execute inferences efficiently on CPU.
    """
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        print(f"[CORE ENGINE] Initializing Whisper Model '{model_size}' on local hardware...")
        
        # Initialize the stateful model interface once during class object allocation
        self.model = WhisperModel(
            model_size_or_path=model_size,
            device=device,
            compute_type=compute_type
        )
        print("[CORE ENGINE] Whisper engine initialization complete. Standing by for stream data.")

    def transcribe_audio_file(self, file_path: str) -> Generator[str, None, None]:
        """
        Accepts a local system path to an audio file, applies Voice Activity Detection (VAD)
        to isolate speech patterns, and returns a generator streaming words/segments out sequentially.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Target audio chunk not found at specified path: {file_path}")

        # vad_filter=True automatically screens out ambient background noise, breaths, and silence gaps
        segments, info = self.model.transcribe(file_path, vad_filter=True)

        for segment in segments:
            # Yield chunks incrementally to mimic a continuous, fluid data stream
            yield segment.text

# Self-contained instantiation block for system verification
if __name__ == "__main__":
    # If executed directly as a standalone file, verify system capability
    engine = TranscriptionEngine()