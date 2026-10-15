from fastapi import APIRouter, Body
from typing import Dict, Any

router = APIRouter(prefix="/dummy", tags=["Dummy Actions"])

@router.post("/update-blocklist")
async def update_blocklist(payload: Dict[str, Any] = Body(...)):
    """Dummy endpoint to update blocklists."""
    customer_id = payload.get("customer_id", "Unknown")
    account_id = payload.get("account_id", "Unknown")
    return {
        "status": "success", 
        "message": f"Customer {customer_id} and Account {account_id} added to internal blocklist.", 
        "action": "blocklist_update"
    }

@router.post("/update-mule-registry")
async def update_mule_registry(payload: Dict[str, Any] = Body(...)):
    """Dummy endpoint to update mule registry."""
    customer_id = payload.get("customer_id", "Unknown")
    account_id = payload.get("account_id", "Unknown")
    return {
        "status": "success", 
        "message": f"Customer {customer_id} and Account {account_id} flagged in Mule Registry.", 
        "action": "mule_registry_update"
    }

@router.post("/update-negative-feeds")
async def update_negative_feeds(payload: Dict[str, Any] = Body(...)):
    """Dummy endpoint to update consortium/negative feeds."""
    customer_id = payload.get("customer_id", "Unknown")
    return {
        "status": "success", 
        "message": f"Customer {customer_id} broadcasted to consortium negative feeds.", 
        "action": "negative_feed_update"
    }

# --- Dummy APIs for Pre-Enrichment UI ---

@router.post("/cibil")
async def dummy_cibil(payload: Dict[str, Any] = Body(default={})):
    return {"source": "Credit Bureaus", "recent_enquiries": 2, "synthetic_identity_risk": "Low"}

@router.get("/npci")
async def dummy_npci():
    return {"source": "NPCI Fraud Registry", "vpa_blacklist_status": "Clean", "cross_bank_alerts": 0}

@router.post("/sanctions")
async def dummy_sanctions(payload: Dict[str, Any] = Body(default={})):
    return {"source": "OFAC / RBI Sanctions", "match_found": False, "pep_status": "None"}

@router.get("/geoip")
async def dummy_geoip():
    return {"source": "MaxMind IP", "country": "India", "is_vpn_or_tor": False, "risk_score": 12.5}

@router.get("/threat")
async def dummy_threat():
    return {"source": "ThreatMetrix", "device_reputation": "Trusted", "bot_probability": 0.05}

@router.get("/telecom")
async def dummy_telecom():
    return {"source": "Telecom API", "recent_sim_swap": False, "days_since_last_port": 450}

@router.post("/cbs")
async def dummy_cbs(payload: Dict[str, Any] = Body(default={})):
    return {"source": "Core Banking System", "account_age_days": 1240, "average_balance": 45000.00, "historical_fraud_flags": 0}

@router.get("/cms")
async def dummy_cms():
    return {"source": "Card Management", "card_status": "Active", "recent_failed_pins": 0}

@router.get("/mobile")
async def dummy_mobile():
    return {"source": "Mobile App Logs", "session_active": True, "biometric_failures_24h": 0}

@router.get("/kyc")
async def dummy_kyc():
    return {"source": "KYC Repository", "kyc_status": "Full KYC", "pan_linked": True}

@router.get("/crm")
async def dummy_crm():
    return {"source": "CRM System", "customer_tier": "Gold", "open_complaints": 0}
