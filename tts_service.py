import os
import base64
import tempfile
import requests
from typing import Optional, Tuple
from pathlib import Path

class TTSService:
    """Text-to-Speech service using Google Cloud TTS API with API Key"""
    
    # Google TTS API endpoint
    API_ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"
    VOICES_ENDPOINT = "https://texttospeech.googleapis.com/v1/voices"
    
    def __init__(self):
        """Initialize the TTS service"""
        # Get API key from environment variable
        # Set GOOGLE_TTS_API_KEY environment variable with your API key
        self.api_key = os.getenv('GOOGLE_TTS_API_KEY')
        
        if not self.api_key:
            print("Warning: GOOGLE_TTS_API_KEY environment variable not set.")
            print("TTS functionality will be disabled. Please set up your Google Cloud API key.")
            print("You can get an API key from: https://console.cloud.google.com/apis/credentials")
        else:
            print("TTS service initialized successfully with API key authentication.")
    
    def _extract_language_code_from_voice(self, voice_name: str) -> str:
        """
        Extract language code from voice name
        
        Voice names follow pattern: "en-US-Chirp-HD-F", "es-ES-Neural2-A", etc.
        Extract the first two parts (language-REGION)
        
        Args:
            voice_name: Google TTS voice name
            
        Returns:
            Language code (e.g., "en-US") or "en-US" as default
        """
        if not voice_name:
            return "en-US"
        
        try:
            # Split by hyphen and take first two parts
            parts = voice_name.split('-')
            if len(parts) >= 2:
                return f"{parts[0]}-{parts[1]}"
            else:
                return "en-US"
        except Exception:
            return "en-US"
    
    def text_to_speech(
        self, 
        text: str, 
        voice_name: Optional[str] = None,
        speaking_rate: float = 1.0,
        pitch: float = 0.0
    ) -> Optional[Tuple[bytes, str]]:
        """
        Convert text to speech using Google TTS API
        
        Args:
            text: Text to convert to speech
            voice_name: Specific voice name (optional, defaults to "en-US-Chirp-HD-F")
                       Language code is automatically extracted from voice name
            speaking_rate: Speaking rate (0.25 to 4.0)
            pitch: Pitch adjustment (-20.0 to 20.0)
            
        Returns:
            Tuple of (audio_bytes, base64_encoded_audio) or None if failed
        """
        if not self.api_key:
            print("TTS API key not available")
            return None
            
        if not text or not text.strip():
            print("Empty text provided for TTS")
            return None
        
        try:
            # Construct the request URL with API key
            url = f"{self.API_ENDPOINT}?key={self.api_key}"
            
            # Use provided voice_name, fallback to env variable, or use default
            effective_voice_name = voice_name or "en-US-Chirp-HD-F"
            
            # Extract language code from voice name
            language_code = self._extract_language_code_from_voice(effective_voice_name)
            
            # Build the voice configuration
            voice_config = {
                "languageCode": language_code,
                "ssmlGender": "NEUTRAL",
                "name": effective_voice_name
            }
            
            # Build the request payload
            payload = {
                "input": {
                    "text": text
                },
                "voice": voice_config,
                "audioConfig": {
                    "audioEncoding": "MP3",
                    "speakingRate": speaking_rate,
                    "pitch": pitch
                }
            }
            
            # Make the API request
            response = requests.post(url, json=payload)
            
            # Check if request was successful
            if response.status_code != 200:
                print(f"TTS API request failed with status {response.status_code}: {response.text}")
                return None
            
            # Parse the response
            response_data = response.json()
            
            # Get the base64 encoded audio from response
            audio_base64 = response_data.get('audioContent')
            if not audio_base64:
                print("No audio content in API response")
                return None
            
            # Decode from base64 to get raw audio bytes
            audio_bytes = base64.b64decode(audio_base64)
            
            return audio_bytes, audio_base64
            
        except requests.exceptions.RequestException as e:
            print(f"Network error in text-to-speech conversion: {e}")
            return None
        except Exception as e:
            print(f"Error in text-to-speech conversion: {e}")
            return None
    
    def save_audio_to_file(self, audio_bytes: bytes, filename: str = None) -> Optional[str]:
        """
        Save audio bytes to a file
        
        Args:
            audio_bytes: Audio content as bytes
            filename: Optional filename, if not provided creates temp file
            
        Returns:
            Path to saved file or None if failed
        """
        try:
            if filename is None:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                    temp_file.write(audio_bytes)
                    return temp_file.name
            else:
                # Save to specified file
                with open(filename, "wb") as audio_file:
                    audio_file.write(audio_bytes)
                return filename
                
        except Exception as e:
            print(f"Error saving audio file: {e}")
            return None
    
    def get_available_voices(self, language_code: str = "en-US") -> list:
        """
        Get list of available voices for a language
        
        Args:
            language_code: Language code to get voices for
            
        Returns:
            List of available voice names
        """
        if not self.api_key:
            return []
            
        try:
            # Construct the request URL with API key and language code
            url = f"{self.VOICES_ENDPOINT}?key={self.api_key}&languageCode={language_code}"
            
            # Make the API request
            response = requests.get(url)
            
            # Check if request was successful
            if response.status_code != 200:
                print(f"Voices API request failed with status {response.status_code}: {response.text}")
                return []
            
            # Parse the response
            response_data = response.json()
            
            voice_list = []
            for voice in response_data.get('voices', []):
                voice_list.append({
                    "name": voice.get('name'),
                    "language_codes": voice.get('languageCodes', []),
                    "ssml_gender": voice.get('ssmlGender', 'NEUTRAL')
                })
            
            return voice_list
            
        except requests.exceptions.RequestException as e:
            print(f"Network error listing voices: {e}")
            return []
        except Exception as e:
            print(f"Error listing voices: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if TTS service is available"""
        return self.api_key is not None and len(self.api_key) > 0

# Create a global instance
tts_service = TTSService()
