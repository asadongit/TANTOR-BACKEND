"""
Banking Pre-Enrichment Node Builder
Acts as a high-speed orchestrator to fetch internal and external data.
"""

import logging
import asyncio
from typing import Callable, Dict, Any, List
from datetime import datetime, timezone

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from .utils import interpolate

logger = logging.getLogger(__name__)

# ==========================================
# Mock API Providers
# ==========================================

async def fetch_cibil_experian(txn: Dict[str, Any]) -> Dict[str, Any]:
    await asyncio.sleep(0.3)
    return {
        "source": "Credit Bureaus",
        "recent_enquiries": 2,
        "synthetic_identity_risk": "Low"
    }

async def fetch_npci_registry(txn: Dict[str, Any]) -> Dict[str, Any]:
    await asyncio.sleep(0.2)
    return {
        "source": "NPCI Fraud Registry",
        "vpa_blacklist_status": "Clean",
        "cross_bank_alerts": 0
    }

async def fetch_sanctions_lists(txn: Dict[str, Any]) -> Dict[str, Any]:
    await asyncio.sleep(0.4)
    return {
        "source": "OFAC / RBI Sanctions",
        "match_found": False,
        "pep_status": "None"
    }

async def fetch_ip_geolocation(txn: Dict[str, Any]) -> Dict[str, Any]:
    await asyncio.sleep(0.1)
    return {
        "source": "MaxMind IP",
        "country": "India",
        "is_vpn_or_tor": False,
        "risk_score": 12.5
    }

async def fetch_threat_metrix(txn: Dict[str, Any]) -> Dict[str, Any]:
    await asyncio.sleep(0.3)
    return {
        "source": "ThreatMetrix",
        "device_reputation": "Trusted",
        "bot_probability": 0.05
    }

async def fetch_telecom_data(txn: Dict[str, Any]) -> Dict[str, Any]:
    await asyncio.sleep(0.5)
    return {
        "source": "Telecom API",
        "recent_sim_swap": False,
        "days_since_last_port": 450
    }

async def fetch_core_banking(txn: Dict[str, Any]) -> Dict[str, Any]:
    await asyncio.sleep(0.2)
    return {
        "source": "Core Banking System",
        "account_age_days": 1240,
        "average_balance": 45000.00,
        "historical_fraud_flags": 0
    }

async def fetch_card_management(txn: Dict[str, Any]) -> Dict[str, Any]:
    await asyncio.sleep(0.2)
    return {
        "source": "Card Management",
        "card_status": "Active",
        "recent_failed_pins": 0
    }

async def fetch_mobile_app_logs(txn: Dict[str, Any]) -> Dict[str, Any]:
    await asyncio.sleep(0.1)
    return {
        "source": "Mobile App Logs",
        "session_active": True,
        "biometric_failures_24h": 0
    }

async def fetch_kyc_repository(txn: Dict[str, Any]) -> Dict[str, Any]:
    await asyncio.sleep(0.3)
    return {
        "source": "KYC Repository",
        "kyc_status": "Full KYC",
        "pan_linked": True
    }

async def fetch_crm_system(txn: Dict[str, Any]) -> Dict[str, Any]:
    await asyncio.sleep(0.2)
    return {
        "source": "CRM System",
        "customer_tier": "Gold",
        "open_complaints": 0
    }

PROVIDER_MAP = {
    # External
    "cibil_experian": fetch_cibil_experian,
    "npci_registry": fetch_npci_registry,
    "sanctions_lists": fetch_sanctions_lists,
    "ip_geolocation": fetch_ip_geolocation,
    "threat_metrix": fetch_threat_metrix,
    "telecom_data": fetch_telecom_data,
    
    # Internal
    "core_banking": fetch_core_banking,
    "card_management": fetch_card_management,
    "mobile_app_logs": fetch_mobile_app_logs,
    "kyc_repository": fetch_kyc_repository,
    "crm_system": fetch_crm_system
}

# ==========================================
# Main Node Builder Entrypoint
# ==========================================
def build_banking_enrichment_node(node_config: NodeConfig) -> Callable:
    """
    Builds a banking enrichment node that concurrently fetches contextual
    data from selected external and internal APIs and appends it to the transaction.
    """
    async def execute_enrichment(state: AgentState) -> dict:
        context = state.get("context", {}) or {}
        
        # 1. Resolve source variable containing the transaction
        source_key = getattr(node_config, "source_key", "canonical_transaction")
        if not source_key: source_key = "canonical_transaction"
        txn_data = context.get(source_key)
        
        if not txn_data:
            logger.warning(f"Enrichment Node {node_config.id}: Could not find '{source_key}' in context.")
            txn_data = {"error": "Missing canonical transaction data"}
        elif isinstance(txn_data, str):
            import json
            try:
                txn_data = json.loads(txn_data)
            except:
                pass
                
        # 2. Get properties from frontend configuration
        selected_external = getattr(node_config, "external_integrations", [])
        selected_internal = getattr(node_config, "internal_integrations", [])
        destination_key = getattr(node_config, "destination_key", "enriched_transaction")
        if not destination_key: destination_key = "enriched_transaction"
        
        # In case they are strings (legacy UI passing CSVs)
        if isinstance(selected_external, str):
            selected_external = [x.strip() for x in selected_external.split(",") if x.strip()]
        if isinstance(selected_internal, str):
            selected_internal = [x.strip() for x in selected_internal.split(",") if x.strip()]
            
        custom_ext = getattr(node_config, "custom_external_apis", [])
        custom_int = getattr(node_config, "custom_internal_apis", [])
        all_custom_apis = []
        if isinstance(custom_ext, list): all_custom_apis.extend(custom_ext)
        if isinstance(custom_int, list): all_custom_apis.extend(custom_int)
            
        all_integrations = selected_external + selected_internal
        if not all_integrations and not all_custom_apis:
            # Fallback test data if backend failed to save the tickboxes
            all_integrations = ["ip_geolocation", "core_banking", "threat_metrix"]
        
        logger.info(f"Enrichment Adapter {node_config.id}: Fetching {len(all_integrations)} legacy integrations and {len(all_custom_apis)} API configurations concurrently.")

        # 3. Build the concurrent tasks
        tasks = []
        task_names = []
        
        # Process legacy list
        for provider_name in all_integrations:
            if provider_name in PROVIDER_MAP:
                tasks.append(PROVIDER_MAP[provider_name](txn_data))
                task_names.append(provider_name)
            else:
                logger.warning(f"Provider {provider_name} is not implemented in the backend.")

        # Add tasks for unified configurations (both custom and mock)
        import urllib.request
        import json
        
        for api_def in all_custom_apis:
            if not isinstance(api_def, dict): continue
            
            provider = api_def.get("provider", "custom")
            
            if provider != "custom" and provider in PROVIDER_MAP:
                # It's a standard mock provider from the unified UI
                tasks.append(PROVIDER_MAP[provider](txn_data))
                name = api_def.get("name") or provider
                task_names.append(name)
            else:
                # It's a custom HTTP API
                async def real_custom_fetch(api_def=api_def):
                    def do_request():
                        try:
                            req_url = api_def.get("url", "")
                            req_method = api_def.get("method", "GET").upper()
                            req_headers = json.loads(api_def.get("headers", "{}"))
                            req_payload = api_def.get("payload", "")
                            
                            data = req_payload.encode('utf-8') if req_method in ["POST", "PUT"] and req_payload else None
                            req = urllib.request.Request(req_url, data=data, headers=req_headers, method=req_method)
                            
                            with urllib.request.urlopen(req, timeout=5) as response:
                                response_body = response.read().decode('utf-8')
                                try:
                                    return json.loads(response_body)
                                except:
                                    return {"response_text": response_body}
                        except Exception as e:
                            return {"error": str(e)}

                    await asyncio.sleep(0) # Yield control
                    result = await asyncio.to_thread(do_request)
                    return {"source": f"Custom API: {api_def.get('name')}", "data": result}
                    
                tasks.append(real_custom_fetch())
                task_names.append(api_def.get("name", "Unknown API"))

        # 4. Execute all API calls concurrently
        enrichment_results = {}
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for name, result in zip(task_names, results):
                if isinstance(result, Exception):
                    logger.error(f"Failed fetching from {name}: {result}")
                    enrichment_results[name] = {"status": "error", "message": str(result)}
                else:
                    enrichment_results[name] = result

        # 5. Merge into the new payload
        enriched_transaction = {}
        if isinstance(txn_data, dict):
            enriched_transaction.update(txn_data)
        else:
            enriched_transaction["raw_input"] = txn_data
            
        enriched_transaction["enrichment_data"] = enrichment_results
        enriched_transaction["enrichment_timestamp"] = datetime.now(timezone.utc).isoformat()

        # 6. Save the enriched data to context
        new_context = {**context, destination_key: enriched_transaction}

        return {
            "messages": [AIMessage(content=f"Enriched transaction with {len(enrichment_results)} data sources.", response_metadata={"active_agent": node_config.id})],
            "active_agent": node_config.id,
            "context": new_context
        }

    return RunnableLambda(execute_enrichment)
