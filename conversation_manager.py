import uuid
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class ConversationMessage:
    """Represents a single message in a conversation"""
    timestamp: str
    audio_filename: str
    transcription: str
    custom_prompt: Optional[str]
    gpt_response: str
    gpt_audio_base64: Optional[str]  # Base64 encoded audio of GPT response
    file_info: dict

@dataclass
class Conversation:
    """Represents a complete conversation"""
    conversation_id: str
    created_at: str
    updated_at: str
    expires_at: str  # ISO format timestamp when conversation expires
    initial_prompt: Optional[str]  # The initial custom prompt for this conversation
    messages: List[ConversationMessage]

class ConversationManager:
    """Manages conversation storage and retrieval"""
    
    def __init__(self, storage_dir: str = "conversations"):
        """Initialize the conversation manager
        
        Args:
            storage_dir: Directory to store conversation files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
    def create_conversation(self, initial_prompt: Optional[str] = None) -> str:
        """Create a new conversation and return its ID
        
        Returns:
            str: Unique conversation ID
        """
        conversation_id = str(uuid.uuid4())
        now = datetime.now()
        timestamp = now.isoformat()
        # Set expiry to 3 hours from creation
        expires_at = (now + timedelta(hours=3)).isoformat()
        
        conversation = Conversation(
            conversation_id=conversation_id,
            created_at=timestamp,
            updated_at=timestamp,
            expires_at=expires_at,
            initial_prompt=initial_prompt,
            messages=[]
        )
        
        self._save_conversation(conversation)
        return conversation_id
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Retrieve a conversation by ID
        
        Args:
            conversation_id: The conversation ID to retrieve
            
        Returns:
            Conversation object if found and not expired, None otherwise
        """
        conversation_file = self.storage_dir / f"{conversation_id}.json"
        
        if not conversation_file.exists():
            return None
            
        try:
            with open(conversation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if conversation has expired
            expires_at_str = data.get('expires_at')
            if expires_at_str:
                expires_at = datetime.fromisoformat(expires_at_str)
                if datetime.now() > expires_at:
                    print(f"Conversation {conversation_id} has expired")
                    # Optionally delete the expired conversation file
                    self.delete_conversation(conversation_id)
                    return None
            else:
                # Handle old conversations without expiry - set expiry to 3 hours from now
                expires_at_str = (datetime.now() + timedelta(hours=3)).isoformat()
                
            # Convert message dictionaries back to ConversationMessage objects
            messages = [ConversationMessage(**msg) for msg in data['messages']]
            
            return Conversation(
                conversation_id=data['conversation_id'],
                created_at=data['created_at'],
                updated_at=data['updated_at'],
                expires_at=expires_at_str,
                initial_prompt=data.get('initial_prompt'),  # Handle old conversations without this field
                messages=messages
            )
        except Exception as e:
            print(f"Error loading conversation {conversation_id}: {e}")
            return None
    
    def add_message(self, conversation_id: str, message: ConversationMessage) -> bool:
        """Add a message to an existing conversation
        
        Args:
            conversation_id: The conversation ID
            message: The message to add
            
        Returns:
            bool: True if successful, False otherwise
        """
        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            return False
            
        conversation.messages.append(message)
        conversation.updated_at = datetime.now().isoformat()
        
        self._save_conversation(conversation)
        return True
    
    def get_conversation_history_for_gpt(self, conversation_id: str) -> List[Dict[str, str]]:
        """Get conversation history formatted for GPT API
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            List of message dictionaries for GPT API
        """
        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            return []
        
        gpt_messages = []
        
        for msg in conversation.messages:
            # Add user message (transcription)
            gpt_messages.append({
                "role": "user",
                "content": f"Audio transcription from {msg.audio_filename}: {msg.transcription}"
            })
            
            # Add assistant response
            gpt_messages.append({
                "role": "assistant", 
                "content": msg.gpt_response
            })
        
        return gpt_messages
    
    def list_conversations(self) -> List[Dict[str, str]]:
        """List all conversations with basic info (excluding expired ones)
        
        Returns:
            List of conversation summaries
        """
        conversations = []
        now = datetime.now()
        
        for conversation_file in self.storage_dir.glob("*.json"):
            try:
                with open(conversation_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check if conversation has expired
                expires_at_str = data.get('expires_at')
                if expires_at_str:
                    expires_at = datetime.fromisoformat(expires_at_str)
                    if now > expires_at:
                        # Skip expired conversations
                        print(f"Skipping expired conversation {data['conversation_id']}")
                        continue
                
                conversations.append({
                    "conversation_id": data['conversation_id'],
                    "created_at": data['created_at'],
                    "updated_at": data['updated_at'],
                    "expires_at": expires_at_str,
                    "message_count": len(data['messages']),
                    "last_audio_file": data['messages'][-1]['audio_filename'] if data['messages'] else None
                })
            except Exception as e:
                print(f"Error reading conversation file {conversation_file}: {e}")
                continue
        
        # Sort by updated_at (most recent first)
        conversations.sort(key=lambda x: x['updated_at'], reverse=True)
        return conversations
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation
        
        Args:
            conversation_id: The conversation ID to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        conversation_file = self.storage_dir / f"{conversation_id}.json"
        
        if not conversation_file.exists():
            return False
            
        try:
            conversation_file.unlink()
            return True
        except Exception as e:
            print(f"Error deleting conversation {conversation_id}: {e}")
            return False
    
    def cleanup_expired_conversations(self) -> int:
        """Delete all expired conversations
        
        Returns:
            int: Number of conversations deleted
        """
        deleted_count = 0
        now = datetime.now()
        
        for conversation_file in self.storage_dir.glob("*.json"):
            try:
                with open(conversation_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check if conversation has expired
                expires_at_str = data.get('expires_at')
                if expires_at_str:
                    expires_at = datetime.fromisoformat(expires_at_str)
                    if now > expires_at:
                        conversation_id = data['conversation_id']
                        if self.delete_conversation(conversation_id):
                            print(f"Deleted expired conversation {conversation_id}")
                            deleted_count += 1
            except Exception as e:
                print(f"Error processing conversation file {conversation_file}: {e}")
                continue
        
        return deleted_count
    
    def _save_conversation(self, conversation: Conversation) -> None:
        """Save a conversation to disk
        
        Args:
            conversation: The conversation to save
        """
        conversation_file = self.storage_dir / f"{conversation.conversation_id}.json"
        
        # Convert to dictionary for JSON serialization
        data = asdict(conversation)
        
        try:
            with open(conversation_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving conversation {conversation.conversation_id}: {e}")
            raise
