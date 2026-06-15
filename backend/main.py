import json
from typing import List, Dict, Any
from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

# 1. Pipeline Component Imports
from backend.database import init_db, SessionLocal, InterviewSession, ResponseAnalytics
from backend.text_analysis import TextAnalysisEngine
from backend.visual_analysis import VisualAnalysisEngine
from backend.signal_fusion import SignalFusionEngine
from backend.star_evaluator import STAREvaluatorEngine
from backend.coaching_engine import LocalCoachingEngine

# Initialize Database Tables on Boot Sequence
init_db()

app = FastAPI(
    title="InsightAI",
    description="High-Performance Multi-Modal Interview Analytics Engine Gateway",
    version="1.0.0"
)

# 2. Global CORS Middleware Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Instantiate Analytical Singletons into Memory
text_engine = TextAnalysisEngine()
vision_engine = VisualAnalysisEngine()
fusion_engine = SignalFusionEngine()
star_evaluator = STAREvaluatorEngine()
coaching_engine = LocalCoachingEngine()


# 4. WebSocket State Management Core
class WebSocketConnectionManager:
    """
    Manages active stateful WebSocket channels, handling client registrations,
    clean safety disconnections, and real-time JSON frame routing.
    """
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[GATEWAY] New real-time telemetry channel established. Active links: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"[GATEWAY] Telemetry channel closed safely. Active links: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)

manager = WebSocketConnectionManager()


# 5. Database Session Lifecycle Injection Guard
def get_db_session():
    """
    Dependency injector yielding a clean database session per request,
    guaranteeing atomic closing even during downstream system failures.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 6. Standard HTTP REST Endpoints
@app.get("/", status_code=status.HTTP_200_OK)
def read_root() -> Dict[str, str]:
    """
    System health check route.
    """
    return {"status": "online", "system": "InsightAI Engine Core Active"}

@app.get("/api/history", status_code=status.HTTP_200_OK)
def get_interview_history(db: Session = Depends(get_db_session)) -> List[Dict[str, Any]]:
    """
    Queries database to fetch archived sessions alongside relational analytics scores.
    """
    sessions = db.query(InterviewSession).order_by(InterviewSession.created_at.desc()).all()
    history_payload = []
    for s in sessions:
        an = s.analytics
        history_payload.append({
            "session_id": s.id,
            "timestamp": s.created_at.isoformat(),
            "question": s.question_text,
            "has_analytics": an is not None,
            "scores": {
                "overall": an.overall_score if an else 0.0,
                "clarity": an.clarity_rating if an else 0.0,
                "pace": an.pace_rating if an else 0.0,
                "visual": an.visual_rating if an else 0.0,
            } if an else None,
            "dominant_emotion": an.dominant_emotion if an else "N/A",
            "summary": an.executive_summary if an else "Processing incomplete."
        })
    return history_payload

@app.delete("/api/session/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_interview_record(session_id: int, db: Session = Depends(get_db_session)):
    """
    Wipes an interview history instance. Cascade rules delete child analytics rows automatically.
    """
    target_record = db.query(InterviewSession).filter(InterviewSession.id == session_id).first()
    if not target_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Deletion failed: Record index '{session_id}' does not exist."
        )
    db.delete(target_record)
    db.commit()
    return None


# 7. Asynchronous Bi-Directional Streaming Route
@app.websocket("/ws/stream")
async def telemetry_stream_endpoint(websocket: WebSocket):
    """
    Continuous pipeline processing incoming granular data tokens over an active socket.
    """
    await manager.connect(websocket)
    session_visual_history = []
    
    try:
        while True:
            raw_data = await websocket.receive_text()
            payload = json.loads(raw_data)
            
            incoming_text = payload.get("text", "")
            duration = float(payload.get("duration", 5.0))
            mock_image_path = payload.get("image_path", "missing_frame.jpg")

            text_metrics = text_engine.analyze_transcript(incoming_text, duration_seconds=duration)
            vision_metrics = vision_engine.analyze_frame(mock_image_path)
            session_visual_history.append(vision_metrics)

            fused_scorecard = fusion_engine.generate_composite_profile(text_metrics, session_visual_history)

            realtime_response = {
                "event": "telemetry_update",
                "current_analytics": text_metrics,
                "current_visual": vision_metrics,
                "running_totals": fused_scorecard
            }
            await manager.send_personal_message(realtime_response, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"[STREAM ERROR] Exception caught in telemetry pipeline: {str(e)}")
        manager.disconnect(websocket)


# 8. Unified Orchestration Session Submission Endpoint
@app.post("/api/session/submit", status_code=status.HTTP_201_CREATED)
def submit_completed_interview(payload: Dict[str, Any], db: Session = Depends(get_db_session)):
    """
    Ingests raw interview feedback components, runs STAR and Coaching evaluations,
    and commits a unified relational dataset down to disk.
    """
    try:
        # Parse payload details
        question = payload.get("question", "Behavioral Interview Question")
        transcript = payload.get("transcript", "")
        overall_score = float(payload.get("overall_score", 70.0))
        clarity_rating = float(payload.get("clarity_rating", 0.7))
        pace_rating = float(payload.get("pace_rating", 0.7))
        visual_rating = float(payload.get("visual_rating", 0.7))
        dominant_emotion = payload.get("dominant_emotion", "neutral")
        
        if not transcript.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Submission rejected: Transcript content cannot be empty."
            )

        # Run Structured STAR Method Evaluation
        star_report = star_evaluator.evaluate_response_structure(transcript)

        # Compile profiles for Local AI Coaching Feedback Generation
        composite_profile = {
            "overall_performance_score": overall_score,
            "dominant_visual_trait": dominant_emotion
        }
        coaching_report = coaching_engine.generate_mock_feedback(composite_profile)
        exec_summary = coaching_report["coach_evaluation"]["executive_summary"]

        # Open transactional context block to secure relational mapping integrity
        new_session = InterviewSession(question_text=question)
        db.add(new_session)
        db.flush() 

        archived_analytics = ResponseAnalytics(
            session_id=new_session.id,
            transcript=transcript,
            overall_score=overall_score,
            clarity_rating=clarity_rating,
            pace_rating=pace_rating,
            visual_rating=visual_rating,
            dominant_emotion=dominant_emotion,
            is_star_compliant=star_report.is_star_compliant,
            executive_summary=exec_summary
        )
        db.add(archived_analytics)
        db.commit()

        return {
            "status": "success",
            "session_id": new_session.id,
            "star_compliance": star_report.is_star_compliant,
            "ai_coaching_summary": exec_summary
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Orchestration failure during transaction layout compile: {str(e)}"
        )