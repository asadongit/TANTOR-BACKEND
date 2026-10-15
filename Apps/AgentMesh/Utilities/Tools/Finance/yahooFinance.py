from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from Apps.AgentMesh.Core.ToolRegistry import registry
from Apps.AgentMesh.Schema.ToolSchema import ToolMeta

# Instantiate the tool
yahoo_news_tool = YahooFinanceNewsTool()

# Define metadata for the tool
yahoo_news_tool_meta = ToolMeta(
	name="yahoo_finance_news",
	display_name="Yahoo Finance News",
	description="Fetches the latest news headlines for a given stock ticker from Yahoo Finance.",
	category="Finance",
)

# Register the tool with the registry
registry.register(yahoo_news_tool, yahoo_news_tool_meta)

# Example usage (for testing, remove or comment out in production)
if __name__ == "__main__":
	print(yahoo_news_tool.invoke("NVDA"))
