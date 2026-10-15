import os
import logging
from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig
from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Core.ToolRegistry import registry
from Apps.AgentMesh.Config import settings

logger = logging.getLogger(__name__)

@tool
def export_as_markdown(content: str, filename: str, config: RunnableConfig = None):
    """
    Saves research notes or agent responses as a professional Markdown (.md) file.
    Saves the file to the agent's virtual filesystem and returns a download link.
    
    Args:
        content: The raw text or markdown content to be saved.
        filename: The desired filename (e.g. 'research_report.md').
        config: System context (injected automatically).
    """
    try:
        if not filename.endswith(".md"):
            filename += ".md"

        # 1. Resolve storage path
        configurable = config.get("configurable", {}) if config else {}
        thread_id = configurable.get("thread_id", "default")
        
        # Use the official exports mount path from Config.py
        root_dir = settings.EXPORTS_MOUNT
        dest_dir = os.path.join(root_dir, thread_id)
        os.makedirs(dest_dir, exist_ok=True)
        
        file_path = os.path.join(dest_dir, filename)

        # 2. Write Markdown content to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"Markdown file exported: {file_path}")
        
        # 3. Return a helpful message with a direct download link
        download_url = f"/exports/{thread_id}/{filename}"
        return f"Successfully generated markdown report. You can download it here: [{filename}]({download_url})"
        
    except Exception as e:
        logger.error(f"Failed to export Markdown: {e}")
        return f"Error generating Markdown file: {str(e)}"

# Register the tool
registry.register(
    export_as_markdown,
    ToolMeta(
        name="export_as_markdown",
        display_name="Export to Markdown",
        description="Convert search notes or research into a downloadable Markdown (.md) file.",
        category="Export",
        package="built-in",
        version="1.0.0"
    )
)
