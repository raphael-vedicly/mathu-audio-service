import os
import tempfile
import shutil
from typing import Optional
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from main import AudioTranscriptionProcessor
from conversation_manager import ConversationManager, ConversationMessage
from tts_service import tts_service
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Audio Transcription API",
    description="Upload audio files for transcription and AI processing using Groq models",
    version="1.0.0"
)

# Response models
class TranscriptionResponse(BaseModel):
    conversation_id: str
    transcription: str
    gpt_response: str
    gpt_audio_base64: Optional[str] = None  # Base64 encoded audio of GPT response
    file_info: dict
    is_new_conversation: bool = False
    tts_enabled: bool = False

class TranscriptionOnlyResponse(BaseModel):
    conversation_id: str
    transcription: str
    file_info: dict
    is_new_conversation: bool = False

class ConversationListResponse(BaseModel):
    conversations: list

class ConversationDetailResponse(BaseModel):
    conversation_id: str
    created_at: str
    updated_at: str
    expires_at: str
    messages: list

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

# Initialize the processor and conversation manager
processor = AudioTranscriptionProcessor()
conversation_manager = ConversationManager()


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe (MP3, WAV, M4A, etc.)"),
    custom_prompt: Optional[str] = Form(None, description="Custom prompt for AI processing (only used for new conversations)"),
    conversation_id: Optional[str] = Form(None, description="Existing conversation ID to continue conversation"),
    enable_tts: bool = Form(True, description="Enable text-to-speech for GPT response"),
    voice_name: Optional[str] = Form(None, description="Voice name for TTS (defaults to 'en-US-Chirp-HD-F')")
):
    """
    Transcribe an audio file and process it with AI
    
    - **file**: Audio file to upload (supports MP3, WAV, M4A, FLAC, etc.)
    - **custom_prompt**: Optional custom prompt for AI processing (default: summarization)
    - **conversation_id**: Optional conversation ID to continue an existing conversation
    - **enable_tts**: Enable text-to-speech for GPT response (default: true)
    - **voice_name**: Voice name for TTS (default: 'en-US-Chirp-HD-F')
    
    Returns transcription and AI-processed response.
    """
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file size (limit to 25MB)
    max_size = 25 * 1024 * 1024  # 25MB
    
    # Supported audio formats
    supported_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.mp4', '.ogg', '.webm'}
    file_extension = os.path.splitext(file.filename.lower())[1]
    
    if file_extension not in supported_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Supported formats: {', '.join(supported_extensions)}"
        )
    
    # Create temporary file
    temp_file = None
    try:
        # Create temporary file with proper extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            # Read file content
            content = await file.read()
            
            # Check file size
            if len(content) > max_size:
                raise HTTPException(
                    status_code=413, 
                    detail=f"File too large. Maximum size is {max_size // (1024*1024)}MB"
                )
            
            # Write content to temporary file
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Handle conversation logic
        is_new_conversation = False
        effective_prompt = custom_prompt  # The prompt that will actually be used
        
        if conversation_id is None:
            # Create new conversation with the provided custom prompt
            conversation_id = conversation_manager.create_conversation(custom_prompt)
            is_new_conversation = True
            conversation_history = []
        else:
            # Get existing conversation
            conversation = conversation_manager.get_conversation(conversation_id)
            if conversation is None:
                raise HTTPException(status_code=404, detail="Conversation not found")
            
            # Use the saved initial prompt from the conversation
            effective_prompt = conversation.initial_prompt
            conversation_history = conversation_manager.get_conversation_history_for_gpt(conversation_id)
            
            # Warn if user provided a custom prompt for existing conversation
            if custom_prompt and custom_prompt != conversation.initial_prompt:
                print(f"Warning: Ignoring custom_prompt '{custom_prompt}' for existing conversation. Using saved prompt: '{effective_prompt}'")
        
        # Transcribe the audio
        transcription = processor.transcribe_audio(temp_file_path)
        if transcription is None:
            raise HTTPException(status_code=500, detail="Transcription failed")
        
        # Process with GPT using conversation history
        print(f"Transcription: {transcription}")
        print(f"Effective prompt: {effective_prompt}")
        print(f"Conversation history: {conversation_history}")
        gpt_response = processor.process_with_gpt(transcription, effective_prompt, conversation_history)
        if gpt_response is None:
            raise HTTPException(status_code=500, detail="GPT processing failed")
        
        # Generate audio from GPT response if TTS is enabled
        gpt_audio_base64 = None
        tts_enabled = False
        
        if enable_tts and tts_service.is_available():
            try:
                print(f"Generating audio from GPT response with voice: {voice_name or 'default (en-US-Chirp-HD-F)'}...")
                tts_result = tts_service.text_to_speech(gpt_response, voice_name=voice_name)
                if tts_result:
                    audio_bytes, gpt_audio_base64 = tts_result
                    tts_enabled = True
                    print(f"Audio generated successfully, size: {len(audio_bytes)} bytes")
                else:
                    print("TTS generation failed")
            except Exception as e:
                print(f"TTS error: {e}")
        elif enable_tts:
            print("TTS requested but service not available")
        else:
            print("TTS disabled by request")
        
        # Prepare file info
        file_info = {
            "filename": file.filename,
            "size_bytes": len(content),
            "format": file_extension,
            "content_type": file.content_type
        }
        
        # Save message to conversation (save the effective prompt used)
        message = ConversationMessage(
            timestamp=datetime.now().isoformat(),
            audio_filename=file.filename,
            transcription=transcription,
            custom_prompt=effective_prompt,
            gpt_response=gpt_response,
            gpt_audio_base64=gpt_audio_base64,
            file_info=file_info
        )
        
        success = conversation_manager.add_message(conversation_id, message)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save conversation")
        
        return TranscriptionResponse(
            conversation_id=conversation_id,
            transcription=transcription,
            gpt_response=gpt_response,
            gpt_audio_base64=gpt_audio_base64,
            file_info=file_info,
            is_new_conversation=is_new_conversation,
            tts_enabled=tts_enabled
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass  # Ignore cleanup errors


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    success = conversation_manager.delete_conversation(conversation_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {"message": "Conversation deleted successfully"}

@app.post("/conversations/cleanup-expired")
async def cleanup_expired_conversations():
    """Clean up all expired conversations (older than 3 hours)"""
    deleted_count = conversation_manager.cleanup_expired_conversations()
    
    return {
        "message": f"Cleanup completed",
        "deleted_count": deleted_count
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
