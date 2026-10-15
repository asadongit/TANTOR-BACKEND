import os
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Singleton manager for sentiment analysis using DistilBERT.
    """
    _instance = None
    _sentiment_model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SentimentAnalyzer, cls).__new__(cls)
        return cls._instance
    
    def get_sentiment_model(self):
        """
        Returns the sentiment analysis pipeline.
        Loads the model once and caches it for subsequent calls.
        """
        if self._sentiment_model is None:
            model_dir = "models/DistilBERT"
            # Check if model is already downloaded
            if not os.path.exists(model_dir):
                logger.info(f"Model not found in {model_dir}, will download to cache.")
            else:
                logger.info(f"Model found in {model_dir}, loading from cache.")
            self._sentiment_model = pipeline("sentiment-analysis", model_kwargs={"cache_dir": model_dir})
            logger.info("Sentiment analysis model loaded successfully")
        return self._sentiment_model
    
    def analyze(self, text):
        model = self.get_sentiment_model()
        return model(text)

# Global instance
sentiment_analyzer = SentimentAnalyzer()