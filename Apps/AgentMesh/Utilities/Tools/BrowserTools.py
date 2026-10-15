import asyncio
import logging
from typing import Optional, Dict, Any
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Core.ToolRegistry import registry

logger = logging.getLogger(__name__)

async def _scrape_page(url: str, wait_time: int = 2000) -> str:
    """
    Helper function to navigate to a URL and extract its text content using Playwright.
    """
    try:
        async with async_playwright() as p:
            # Launch browser (chromium is generally fastest)
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            # Set a common user agent to avoid basic bot detection
            await page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
            })

            # Navigate with timeout
            logger.info(f"Navigating to {url}...")
            await page.goto(url, wait_until="networkidle", timeout=30000)
            
            # Wait a bit for JS-heavy sites to settle
            if wait_time > 0:
                await asyncio.sleep(wait_time / 1000)

            # Get content
            content = await page.content()
            await browser.close()

            # Clean content with BeautifulSoup
            soup = BeautifulSoup(content, "html.parser")
            
            # Remove scripts, styles, and other non-informative tags
            for script_or_style in soup(["script", "style", "header", "footer", "nav", "aside"]):
                script_or_style.decompose()

            # Get text and clean whitespace
            text = soup.get_text(separator="\n")
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)

            return text[:10000]  # Return first 10k chars to avoid token limits
            
    except Exception as e:
        logger.error(f"Error scraping {url}: {str(e)}")
        return f"Error: Failed to access the website. {str(e)}"

@tool
async def browse_website(url: str) -> str:
    """
    Visit a website URL and extract its main text content. 
    Use this when you need detailed information from a specific webpage found during search.
    """
    return await _scrape_page(url)

# Register the tool in the central registry
registry.register(
    browse_website,
    ToolMeta(
        name="browse_website",
        display_name="Browser (Playwright)",
        description="Visit any URL to read its full content. Highly effective for deep research and reading articles.",
        category="Search",
        package="built-in",
        version="1.1.0",
        ai_visible_fields=["url"]
    )
)

logger.info("Playwright Browser tool registered successfully.")
