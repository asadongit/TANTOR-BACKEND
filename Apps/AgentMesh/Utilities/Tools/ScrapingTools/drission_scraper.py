import asyncio
import logging
import json
from typing import Optional, List, Dict, Any
from DrissionPage import ChromiumPage, ChromiumOptions
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Core.ToolRegistry import registry

logger = logging.getLogger(__name__)

def _get_page(headless: bool = True):
    """Helper to initialize DrissionPage with stealth settings."""
    co = ChromiumOptions()
    co.set_argument('--no-sandbox')
    co.set_argument('--disable-gpu')
    
    # Speed Optimization: Disable images to load pages much faster
    co.set_pref('profile.managed_default_content_settings.images', 2) 
    
    if headless:
        co.headless()
    return ChromiumPage(co)

@tool
async def drission_smart_scrape(url: str, wait_for_selector: Optional[str] = None, config: RunnableConfig = None) -> str:
    """
    Generic High-Stealth Scraper using DrissionPage.
    Works for ANY website, including those with heavy anti-bot protections (Amazon, Flipkart, LinkedIn, etc.)
    - url: The full web address to scrape.
    - wait_for_selector: (Optional) CSS selector to wait for before extracting data.
    """
    try:
        page = _get_page(headless=True)
        logger.info(f"DrissionPage: Navigating to {url}")
        
        page.get(url)
        
        # Optional wait
        if wait_for_selector:
            page.wait.ele_displayed(wait_for_selector, timeout=10)
        else:
            # Smart wait for any content body
            page.wait.ele_displayed('tag:body', timeout=5)
            # Extra second for dynamic content (JS)
            page.wait(2)
            
        # Extract main content
        # We try to get clean text but we can also return HTML if needed
        # For AI agents, clean text is usually better
        content = page.html
        
        # Heuristic: Extract visible text to avoid HTML noise
        # DrissionPage has a very clean 'text' attribute for the whole page
        clean_text = page.run_js("return document.body.innerText")
        
        page.quit()
        
        if not clean_text:
            return "Failed to extract text content."
            
        return clean_text[:15000] # Return a large chunk for the Agent to process

    except Exception as e:
        logger.error(f"DrissionPage smart scrape failed: {str(e)}")
        if 'page' in locals(): page.quit()
        return f"Error: {str(e)}"

@tool
async def drission_site_search(site_url: str, query: str, config: RunnableConfig = None) -> str:
    """
    Universal Site Search tool.
    Goes to a website, finds the search bar, enters your query, and returns the results.
    Works for Amazon, Flipkart, eBay, and most e-commerce sites.
    - site_url: The homepage of the site (e.g., 'https://www.amazon.in')
    - query: What to look for.
    """
    try:
        page = _get_page(headless=True) # Changed to True to keep it hidden
        page.get(site_url)
        
        # Intelligent search bar discovery
        # Most search bars are input[type="text"] or have "search" in their ID/name
        search_bar = page.ele('tag:input@type=text') or page.ele('@placeholder*search') or page.ele('@name*search')
        
        if not search_bar:
            page.quit()
            return f"Could not find a search bar on {site_url} automatically."
            
        search_bar.input(query)
        search_bar.input('\n') # Press Enter
        
        page.wait(3) # Wait for results
        
        # Extract snippets of the results
        results = page.run_js("return document.body.innerText")
        page.quit()
        
        return results[:10000]

    except Exception as e:
        logger.error(f"DrissionPage site search failed: {str(e)}")
        if 'page' in locals(): page.quit()
        return f"Error: {str(e)}"

def register_drission_tools():
    logger.info("Registering DrissionPage Generic tools...")
    registry.register(
        drission_smart_scrape,
        ToolMeta(
            name="drission_smart_scrape",
            display_name="Smart Browser (Stealth)",
            description="Visits any URL using a stealth browser. Perfect for Amazon, Flipkart, and sites that block normal bots.",
            category="Scraping",
            package="external",
            version="1.1.0",
            ai_visible_fields=["url", "wait_for_selector"]
        )
    )
    registry.register(
        drission_site_search,
        ToolMeta(
            name="drission_site_search",
            display_name="Universal Site Search",
            description="Navigates to any site, finds the search bar, and searches for your query.",
            category="Scraping",
            package="external",
            version="1.1.0",
            ai_visible_fields=["site_url", "query"]
        )
    )
    logger.info("DrissionPage Generic tools registered successfully.")

if __name__ == "__main__":
    pass
else:
    register_drission_tools()
