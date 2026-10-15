import os
from typing import Optional
import whisper
import logging
import asyncio
import tempfile
try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip

from Apps.AgentMesh.Core.Services.IngestionService import IngestionService

logger = logging.getLogger(__name__)

class AudioIngestionService:
    """
    Ingests audio and video files by transcribing them with Whisper and storing them
    using the standard RAG ingestion pipeline.
    """
    
    _model = None

    @classmethod
    def load_model(cls, model_name: str = "base"):
        """Loads Whisper model into memory if not already loaded."""
        if cls._model is None:
            logger.info(f"Loading Whisper model: {model_name}")
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
        model_name: str = "base",
        source_type: str = "audio",
        original_filename: Optional[str] = None
    ):
        """
        1. Transcribe audio using Whisper.
        2. Ingest the resulting text into the vector store.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # 1. Transcribe
            model = cls.load_model(model_name)
            logger.info(f"Starting Whisper transcription for: {file_path} using model: {model_name}")
            
            # Transcription is compute intensive and synchronous, run in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: model.transcribe(file_path))
            
            transcript = result.get("text", "")
            language = result.get("language", "unknown")
            
            if not transcript.strip():
                logger.warning(f"No speech detected in file: {file_path}")
                return 0

            # 2. Prepare metadata
            filename = original_filename or os.path.basename(file_path)
            metadata = {
                "source": filename,
                "filename": filename,
                "type": source_type,
                "language": language,
                "transcription_model": model_name
            }

            # 3. Ingest via standard IngestionService
            logger.info(f"Ingesting transcript from {source_type}: {filename} (Language: {language})")
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
            logger.error(f"Error during media ingestion for {file_path}: {e}")
            raise

    @classmethod
    async def ingest_video(
        cls, 
        file_path: str, 
        collection_name: str, 
        vector_store_type: str = "chroma",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        chunking_strategy: str = "recursive",
        model_name: str = "base"
    ):
        """Extracts audio from video and then transcribes/ingests."""
        audio_temp_path = None
        try:
            logger.info(f"Extracting audio from video: {file_path}")
            
            def extract_audio_task():
                nonlocal audio_temp_path
                video = VideoFileClip(file_path)
                try:
                    if video.audio is None:
                        raise ValueError("Video file has no audio track")
                    
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                        audio_temp_path = tmp.name
                    
                    video.audio.write_audiofile(audio_temp_path, logger=None)
                finally:
                    video.close()
                return audio_temp_path

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, extract_audio_task)

            # Transcribe the extracted audio
            count = await cls.ingest_audio(
                file_path=audio_temp_path,
                collection_name=collection_name,
                vector_store_type=vector_store_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunking_strategy=chunking_strategy,
                model_name=model_name,
                source_type="video",
                original_filename=os.path.basename(file_path)
            )

            # Cleanup temp audio
            if audio_temp_path and os.path.exists(audio_temp_path):
                os.unlink(audio_temp_path)
            
            return count
        except Exception as e:
            logger.error(f"Failed to process video: {e}")
            if audio_temp_path and os.path.exists(audio_temp_path):
                os.unlink(audio_temp_path)
            raise
