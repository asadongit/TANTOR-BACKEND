from typing import Dict, Optional
from pydantic import BaseModel, Field


class SubApiConfig(BaseModel):
    """Configuration for an individual sub-API call in multi-API node"""

    name: str = Field("Sub API", description="Name/Label of this sub-API call")
    description: Optional[str] = Field(
        None, description="Detailed explanation of what this API call does"
    )
    enabled: bool = Field(True, description="Whether this API call is executed")
    connection_id: Optional[str] = Field(
        None, description="ID of the saved connection profile in the registry"
    )
    api_protocol: str = Field("REST", description="Protocol type: REST, SOAP, GraphQL")
    api_method: str = "GET"
    api_url: Optional[str] = None
    api_headers: Dict[str, str] = Field(default_factory=dict, repr=False)
    api_params: Dict[str, str] = Field(default_factory=dict, repr=False)
    api_body: Optional[str] = Field(None, repr=False)
    api_auth_type: str = "None"
    api_auth_config: Dict[str, str] = Field(default_factory=dict, repr=False)
    api_response_format: str = "json"
    use_curl: bool = Field(
        False,
        description="Whether to parse and execute a raw cURL command instead of standard fields",
    )
    curl_command: Optional[str] = Field(
        None, description="The raw cURL command string", repr=False
    )
    max_retries: int = Field(1, description="Max retry attempts for this sub-API")
    retry_delay: float = Field(
        2.0, description="Delay between retries in seconds for this sub-API"
    )
