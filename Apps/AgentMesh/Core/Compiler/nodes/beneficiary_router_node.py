import logging
import asyncio
from typing import Callable, Dict, Any
import random

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from .utils import get_production_data

logger = logging.getLogger(__name__)

def build_beneficiary_risk_node(node_config: NodeConfig) -> Callable:
    """
    Builds the Beneficiary Risk Router Node.
    This node fetches risk indicators for the destination account (mule velocity, account age, sanctions)
    and routes the transaction to one of three edges:
    - known_fraudster
    - new_beneficiary
    - safe
    """
    async def beneficiary_risk_node(state: AgentState):
        logger.info(f"Beneficiary Router {node_config.id}: Initiating risk evaluation...")
        
        # 1. Extract context data
        full_scope = get_production_data(state)
        context_key = getattr(node_config, "context_key", "enriched_transaction")
        txn_data = full_scope.get(context_key, {})
        
        # Extract beneficiary info from the canonical schema based on UI configuration
        beneficiary_field = getattr(node_config, "beneficiary_field", "destination_account")
        dest_acct = txn_data.get(beneficiary_field, "UNKNOWN")
        dest_bank = txn_data.get("destination_bank", "UNKNOWN")
        
        logger.info(f"Beneficiary Router {node_config.id}: Analyzing Account {dest_acct} at {dest_bank}")
        
        # 2. Determine which checks to run based on UI properties
        active_checks = getattr(node_config, "risk_checks", [])
        if not active_checks:
            # Fallback for testing if UI hasn't saved yet
            active_checks = ['mule_detection', 'account_age', 'sanctions_list', 'relationship_graph']
            
        # 3. Concurrent mock data gathering for the beneficiary
        async def fetch_mule_score():
            await asyncio.sleep(0.3)
            # Simulate high mule velocity for specific test accounts
            score = 0.95 if dest_acct.endswith("999") else round(random.uniform(0.01, 0.2), 2)
            return {"mule_probability": score, "recent_deposits": random.randint(1, 50)}
            
        async def fetch_account_age():
            await asyncio.sleep(0.1)
            # Simulate fresh accounts for specific test strings
            age_days = 0.5 if "new" in dest_acct.lower() else random.randint(100, 3000)
            return {"account_age_days": age_days}
            
        async def fetch_sanctions():
            await asyncio.sleep(0.4)
            # Simulate a hit on OFAC
            is_hit = dest_acct == "000000000"
            return {"sanctions_hit": is_hit, "list_name": "OFAC SDN" if is_hit else None}
            
        async def fetch_relationship():
            await asyncio.sleep(0.2)
            # Simulate first-time vs returning
            history = 0 if "new" in dest_acct.lower() else random.randint(5, 100)
            return {"historical_successful_txns": history}

        # Build task list based on active checks
        tasks = []
        if 'mule_detection' in active_checks: tasks.append(fetch_mule_score())
        if 'account_age' in active_checks: tasks.append(fetch_account_age())
        if 'sanctions_list' in active_checks: tasks.append(fetch_sanctions())
        if 'relationship_graph' in active_checks: tasks.append(fetch_relationship())

        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge results into a single risk profile
        risk_profile = {}
        for res in results:
            if isinstance(res, dict):
                risk_profile.update(res)
                
        logger.info(f"Beneficiary Router {node_config.id}: Risk Profile Built -> {risk_profile}")

        # --- 4. Decide Routing Intent ---
        intent = "safe"
        
        # Condition 1: Known Fraudster (Strictly Deterministic Matches Only)
        if risk_profile.get("sanctions_hit") is True:
            intent = "known_fraudster"
            logger.warning(f"Beneficiary Router {node_config.id}: Routing to KNOWN FRAUDSTER!")
            
        # Condition 2: New Beneficiary (Based on User-Defined Rule in UI)
        elif intent == "safe":
            rule_str = getattr(node_config, "new_beneficiary_rule", "account_age_days < 1 or historical_successful_txns == 0")
            
            # Safe evaluator that defaults missing variables to 999 (to fail-safe age < 1 rules)
            class SafeDict(dict):
                def __missing__(self, key):
                    return 999
                    
            try:
                is_new = eval(rule_str, {"__builtins__": {}}, SafeDict(risk_profile))
            except Exception as e:
                logger.error(f"Beneficiary Router {node_config.id}: Error evaluating new_beneficiary_rule '{rule_str}': {e}")
                is_new = (risk_profile.get("account_age_days", 999) < 1 or risk_profile.get("historical_successful_txns", 999) == 0)

            if is_new:
                intent = "new_beneficiary"
                logger.warning(f"Beneficiary Router {node_config.id}: Routing to NEW BENEFICIARY (Step-Up Required).")
            
        else:
            logger.info(f"Beneficiary Router {node_config.id}: Routing to SAFE/TRUSTED.")

        # 5. Shape the payload based on the routing intent
        clean_context = {k: v for k, v in full_scope.items() if k not in ("msg", "data")}
        clean_context["beneficiary_risk_profile"] = risk_profile

        # Default payload behavior
        payload_to_pass = txn_data

        if intent == "new_beneficiary":
            # Pass ONLY the beneficiary details for the Registration Node API to consume
            payload_to_pass = {
                "destination_account": dest_acct,
                "destination_bank": dest_bank,
                "routing_reason": "First-time transfer or account age < 1 day",
                "risk_profile": risk_profile
            }
            if node_config.get("registration_api_endpoint"):
                payload_to_pass["custom_registration_endpoint"] = node_config.get("registration_api_endpoint")
                
            logger.info(f"Beneficiary Router {node_config.id}: Shaping payload to ONLY include Beneficiary details for registration.")
            
        elif intent == "known_fraudster":
            # Pass specialized fraud alert payload
            payload_to_pass = {
                "alert_type": "HIGH_RISK_BENEFICIARY",
                "destination_account": dest_acct,
                "reason": "Deterministic match on Sanctions/Watchlist Registry",
                "original_transaction": txn_data
            }
            if node_config.get("fraud_api_endpoint"):
                payload_to_pass["custom_verification_endpoint"] = node_config.get("fraud_api_endpoint")
            if node_config.get("fraud_additional_apis"):
                payload_to_pass["custom_additional_endpoints"] = node_config.get("fraud_additional_apis")

        # Send the shaped payload down the selected route
        clean_context["data"] = payload_to_pass

        # Store beneficiary risk decision in its own key — do NOT overwrite latest_api_result
        # latest_api_result is RESERVED for the ML API node (ml_scoring_engine)
        beneficiary_risk_result = {
            "predictions": [
                {
                    "predictions_1": 1 if intent == "known_fraudster" else 0
                }
            ]
        }
        clean_context["beneficiary_risk_result"] = beneficiary_risk_result
        
        # Set decision keys on transaction payload in context
        txn_key = "enriched_transaction" if "enriched_transaction" in clean_context else "raw_transaction"
        if txn_key in clean_context:
            clean_context[txn_key]["rulebase_fraud_decision"] = (intent == "known_fraudster")
            clean_context[txn_key]["is_fraud"] = (intent == "known_fraudster")

        from langchain_core.messages import AIMessage
        msg_content = f"Beneficiary Router execution completed.\nDecision Intent: **{intent}**"
        
        return {
            "messages": [AIMessage(content=msg_content, response_metadata={"active_agent": node_config.id})],
            "intent": intent, # This string physically routes the graph edge
            "active_agent": node_config.id,
            "context": clean_context
        }

    return beneficiary_risk_node
