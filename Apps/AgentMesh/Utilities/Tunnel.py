import ngrok
import logging
import os
from Apps.AgentMesh.Config import settings

logger = logging.getLogger(__name__)

class TunnelManager:
    """
    Manages the ngrok tunnel for external webhook access.
    """
    _listener = None

    @classmethod
    async def start(cls, port: int = 8000):
        if cls._listener:
            return cls._listener.url()

        authtoken = settings.NGROK_AUTHTOKEN or os.environ.get("NGROK_AUTHTOKEN")
        if not authtoken:
            logger.warning("NGROK_AUTHTOKEN not found. Tunnel cannot be started.")
            return None

        try:
            logger.info("Starting ngrok tunnel...")
            cls._listener = await ngrok.forward(port, authtoken=authtoken)
            
            url = cls._listener.url()
            logger.info(f"Ngrok tunnel established: {url}")
            print(f"\n" + "="*50)
            print(f"🚀 PUBLIC WEBHOOK ACTIVE")
            print(f"🔗 URL: {url}")
            print(f"👉 Use this in MS Power Automate/Excel")
            print("="*50 + "\n")
            
            return url
        except Exception as e:
            logger.error(f"Failed to start ngrok tunnel: {e}")
            return None

    @classmethod
    async def stop(cls):
        if cls._listener:
            logger.info("Closing ngrok tunnel...")
            await ngrok.disconnect()
            cls._listener = None
