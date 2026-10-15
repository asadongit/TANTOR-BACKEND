import asyncio
import logging
from typing import List, Optional
from crawlee.crawlers import PlaywrightCrawler, PlaywrightCrawlingContext
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Core.ToolRegistry import registry

logger = logging.getLogger(__name__)

@tool
async def crawlee_smart_crawl(url: str, max_pages: int = 1, config: RunnableConfig = None) -> str:
    """
    Advanced Generic Crawler using Crawlee.
    Use this for deep scraping or when you need to follow links on a page.
    - url: The starting URL.
    - max_pages: How many pages to follow (default 1).
    """
    results = []
    
    try:
        # Note: We use the correct import path found via inspection
        crawler = PlaywrightCrawler(
            max_requests_per_crawl=max_pages,
            # Crawlee's default PlaywrightCrawler has high anti-detection
        )

        @crawler.router.default_handler
        async def request_handler(context: PlaywrightCrawlingContext) -> None:
            context.log.info(f"Crawlee: Scraping {context.request.url}")
            
            # Extract basic text and title
            title = await context.page.title()
            # Extract all text from body
            content = await context.page.evaluate("() => document.body.innerText")
            
            results.append(f"Source: {context.request.url}\nTitle: {title}\nContent:\n{content[:5000]}")

        await crawler.run([url])
        
        if not results:
            return "Crawl finished but no content was extracted."
            
        return "\n\n" + ("="*30) + "\n\n".join(results)
        
    except Exception as e:
        logger.error(f"Crawlee smart crawl failed: {str(e)}")
        return f"Error: {str(e)}"

def register_crawlee_tools():
    logger.info("Registering Crawlee Generic tools...")
    registry.register(
        crawlee_smart_crawl,
        ToolMeta(
            name="crawlee_smart_crawl",
            display_name="Crawl Site (Crawlee)",
            description="Deep crawl any website. Can follow links and extract large amounts of data.",
            category="Scraping",
            package="external",
            version="1.1.0",
            ai_visible_fields=["url", "max_pages"]
        )
    )
    logger.info("Crawlee Generic tools registered successfully.")

if __name__ == "__main__":
    pass
else:
    register_crawlee_tools()
