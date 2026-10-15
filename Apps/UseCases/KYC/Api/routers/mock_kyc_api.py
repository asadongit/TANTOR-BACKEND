from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import json
import asyncio
import random
from pathlib import Path

router = APIRouter(prefix="/mock-kyc", tags=["Mock KYC & Identity APIs"])

JSON_PATH = Path(r"C:\Users\Translab\Downloads\mock_indian_customers.json")
CKYC_JSON_PATH = Path(r"C:\Users\Translab\Downloads\mock_ckyc_registry.json")

# ── Fallback In-Memory Datasets ─────────────────────────────────────────────
FALLBACK_CUSTOMERS = [
    {
        "full_name": "Vijay Vithal Mallya",
        "aadhaar_number": "598367190433",
        "pan_number": "ABCDE1234F",
        "passport_number": "Z1234567",
        "driving_license": "DL1420110012345",
        "voter_id": "ABC1234567",
        "date_of_birth": "1955-12-18",
        "gender": "MALE",
        "address": "UB Tower, Vittal Mallya Road, Bengaluru, Karnataka 560001",
        "nationality": "INDIAN"
    },
    {
        "full_name": "Pritam Sahu",
        "aadhaar_number": "483920194827",
        "pan_number": "PQRST5678U",
        "passport_number": "M9876543",
        "driving_license": "DL0420180098765",
        "voter_id": "XYZ9876543",
        "date_of_birth": "1975-03-24",
        "gender": "MALE",
        "address": "H No 458, Ward No 39, Dipra Para, Durg, Chhattisgarh 491001",
        "nationality": "INDIAN"
    },
    {
        "full_name": "Aniket Bandral",
        "aadhaar_number": "220050333928",
        "pan_number": "ABCDE9999Z",
        "passport_number": "Z9999999",
        "driving_license": "DL9920200099999",
        "voter_id": "XYZ9999999",
        "date_of_birth": "2000-08-14",
        "gender": "MALE",
        "address": "S/O: Aniket Bandral, Village Plassi, Plasi, Kathua, Jammu an Kashmir, 184201",
        "nationality": "INDIAN"
    }
]

FALLBACK_CKYC = [
    {
        "kin_number": "900123456789",
        "full_name": "Vijay Vithal Mallya",
        "date_of_birth": "1955-12-18",
        "pan": "ABCDE1234F",
        "aadhaar": "598367190433",
        "passport_number": "Z1234567",
        "nationality": "INDIAN",
        "kyc_status": "VERIFIED"
    },
    {
        "kin_number": "900987654321",
        "full_name": "Pritam Sahu",
        "date_of_birth": "1975-03-24",
        "pan": "PQRST5678U",
        "aadhaar": "483920194827",
        "passport_number": "M9876543",
        "nationality": "INDIAN",
        "kyc_status": "VERIFIED"
    },
    {
        "kin_number": "900220050333",
        "full_name": "Aniket Bandral",
        "date_of_birth": "2000-08-14",
        "pan": "ABCDE9999Z",
        "aadhaar": "220050333928",
        "passport_number": "Z9999999",
        "nationality": "INDIAN",
        "kyc_status": "VERIFIED"
    }
]

def load_ckyc_dataset():
    if CKYC_JSON_PATH.exists():
        try:
            with open(CKYC_JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return FALLBACK_CKYC

# ── CKYC Registry Lookup ──────────────────────────────────────────────────


class CKYCLookupRequest(BaseModel):
    document_id: str = ""
    document_type: str = ""
    name: Optional[str] = None

@router.post("/ckyc/lookup")
def ckyc_lookup(req: CKYCLookupRequest):
    """
    Mock Central KYC Registry (CERSAI) API.
    Looks up a customer by document_id and document_type.
    Falls back to name search if document_id not found.
    """
    if not req.document_id and not req.name:
        return {
            "status": "success",
            "ckyc_status": "NOT_FOUND",
            "kin_number": None,
            "existing_record": None,
            "message": "No identifiers provided for CKYC lookup."
        }

    data = load_ckyc_dataset()

    key_map = {
        "PAN_CARD": "pan",
        "AADHAAR_CARD": "aadhaar",
        "PASSPORT": "passport_number",
        "DRIVERS_LICENSE": None,
        "VOTER_ID": None
    }

    search_key = key_map.get(req.document_type.upper())
    matched = None

    if req.document_id and search_key:
        req_doc = req.document_id.replace(" ", "").strip().upper()
        for record in data:
            rec_val = str(record.get(search_key, "")).replace(" ", "").strip().upper()
            if rec_val and rec_val == req_doc:
                matched = record
                break

    if not matched and req.name:
        search_name = req.name.strip().upper()
        for record in data:
            if search_name in str(record.get("full_name", "")).strip().upper():
                matched = record
                break

    if matched:
        return {
            "status": "success",
            "ckyc_status": "FOUND",
            "kin_number": matched.get("kin_number"),
            "existing_record": matched
        }
    else:
        return {
            "status": "success",
            "ckyc_status": "NOT_FOUND",
            "kin_number": None,
            "existing_record": None,
            "message": "No existing CKYC record found for the given identifiers."
        }

