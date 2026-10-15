import os
from typing import Optional
import whisper
import logging
import asyncio

from Apps.AgentMesh.Core.Services.IngestionService import IngestionService

logger = logging.getLogger(__name__)

class AudioToTextIngestion:
    """
    Ingests audio files by transcribing them with Whisper and storing them
    using the standard RAG ingestion pipeline.
    """
    
    _model = None

    @classmethod
    def load_model(cls, model_name: str = "base"):
        """Loads Whisper model into memory if not already loaded."""
        if cls._model is None:
            logger.info(f"Loading Whisper model: {model_name}")
            # Ensure model is downloaded/loaded. This might take time first time.
            cls._model = whisper.load_model(model_name)
        return cls._model

    @classmethod
    async def ingest_audio(
        cls, 
        file_path: str, 
        collection_name: str, 
        vector_store_type: str = "chroma",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        chunking_strategy: str = "recursive",
        model_name: str = "base"
    ):
        """
        1. Transcribe audio using Whisper.
        2. Ingest the resulting text into the vector store.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        try:
            # 1. Transcribe
            # whisper.transcribe is a synchronous call, run it in a thread to keep async happy
            model = cls.load_model(model_name)
            logger.info(f"Starting Whisper transcription for: {file_path}")
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: model.transcribe(file_path))
            
            transcript = result.get("text", "")
            language = result.get("language", "unknown")
            
            if not transcript.strip():
                logger.warning(f"No speech detected in audio file: {file_path}")
                return 0

            # 2. Prepare metadata
            filename = os.path.basename(file_path)
            metadata = {
                "source": file_path,
                "filename": filename,
                "type": "audio",
                "language": language,
                "transcription_model": model_name
            }

            # 3. Ingest via standard IngestionService
            logger.info(f"Ingesting transcript from audio: {filename} (Language: {language})")
            count = await IngestionService.ingest_text(
                text=transcript,
                collection_name=collection_name,
                metadata=metadata,
                vector_store_type=vector_store_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunking_strategy=chunking_strategy
            )
            
            return count
        except Exception as e:
            logger.error(f"Error during audio ingestion for {file_path}: {e}")
            raise
