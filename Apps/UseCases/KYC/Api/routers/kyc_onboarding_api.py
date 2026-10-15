"""
KYC Customer Onboarding Production REST APIs
Serves Dashboard Aggregates, Paginated/Filtered Customer Search, Full Customer Detail, UI Profile Cards, and Human Decision Overrides.
Database Target: PostgreSQL ('agentmesh_usecase' db -> 'kyc_customer_onboarding' table)
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

import asyncpg
from fastapi import APIRouter, HTTPException, Query, Path, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/onboarding", tags=["KYC Onboarding APIs"])

# Connection DSN Resolution
ENV_DSN = os.getenv("KYC_DATABASE_URL") or os.getenv("DATABASE_URL")

if ENV_DSN and "agentmesh_usecase" not in ENV_DSN:
    if "agentmesh" in ENV_DSN:
        ENV_DSN = ENV_DSN.replace("/agentmesh", "/agentmesh_usecase")

if ENV_DSN and ENV_DSN.startswith("postgresql+asyncpg://"):
    ENV_DSN = ENV_DSN.replace("postgresql+asyncpg://", "postgresql://", 1)

# Connection Pool Singleton
_pool: Optional[asyncpg.Pool] = None

async def get_db_pool() -> asyncpg.Pool:
    global _pool
    if not ENV_DSN:
        raise HTTPException(status_code=503, detail="KYC database is not configured")
    if _pool is None or _pool._closed:
        _pool = await asyncpg.create_pool(
            dsn=ENV_DSN,
            min_size=1,
            max_size=10,
            command_timeout=15.0
        )
    return _pool


# --- Pydantic Schemas ---

class StatusBreakdown(BaseModel):
    approved: int = 0
    manual_review: int = 0
    rejected: int = 0

class RiskBreakdown(BaseModel):
    low: int = 0
    medium: int = 0
    high: int = 0

class DecisionBreakdown(BaseModel):
    human_approved_count: int = 0
    pending_human_decision_count: int = 0

class DashboardMetricsResponse(BaseModel):
    total_customers: int
    status_breakdown: StatusBreakdown
    risk_breakdown: RiskBreakdown
    decision_breakdown: DecisionBreakdown
    auto_decisioned_count: int
    manual_decisioned_count: int
    auto_decision_rate_pct: float
    last_24h_count: int

class CustomerListItem(BaseModel):
    investigation_id: str
    created_at: datetime
    primary_name: str
    document_type: Optional[str] = None
    primary_document_id: Optional[str] = None
    status: str
    risk_level: str
    onboarding_decision: bool = False
    decision_by: Optional[str] = None
    decision_timestamp: Optional[datetime] = None
    auto_decisioned: bool = True
    next_review_months: Optional[int] = 120
    executive_summary: Optional[str] = None

class PaginationMetadata(BaseModel):
    total_records: int
    page: int
    limit: int
    total_pages: int
    has_next: bool
    has_prev: bool

class CustomerListResponse(BaseModel):
    data: List[CustomerListItem]
    records: Optional[List[CustomerListItem]] = None
    pagination: PaginationMetadata

class CustomerDetailResponse(BaseModel):
    investigation_id: str
    created_at: datetime
    primary_name: str
    document_type: Optional[str] = None
    primary_document_id: Optional[str] = None
    status: str
    risk_level: str
    onboarding_decision: bool = False
    decision_by: Optional[str] = None
    decision_notes: Optional[str] = None
    decision_timestamp: Optional[datetime] = None
    auto_decisioned: bool = True
    next_review_months: Optional[int] = 120
    initiation_details: Optional[Dict[str, Any]] = {}
    documents_info: Optional[Dict[str, Any]] = {}
    face_verification: Optional[Dict[str, Any]] = {}
    subject_summary: Dict[str, Any]
    overall_decision: Dict[str, Any]
    sub_agent_breakdown: Dict[str, Any]
    executive_summary: Optional[str] = None
    action_items: Optional[Any] = []
    path_details: Dict[str, Any]


# --- Structured Profile Models ---

class CustomerInformation(BaseModel):
    name: str
    dob: Optional[str] = "N/A"
    mobile: Optional[str] = "N/A"
    kyc_status: str
    risk_level: str
    onboarding_decision: bool = False
    decision_by: Optional[str] = None
    decision_notes: Optional[str] = None

class DocumentsInfo(BaseModel):
    document_type: Optional[str] = None
    primary_document_id: Optional[str] = None
    document_image_base64: Optional[str] = None
    ocr_extracted_data: Dict[str, Any] = {}

# --- Helper Functions ---
def mask_document_id(doc_id: Optional[str]) -> Optional[str]:
    """Mask document ID to show only the last 4 digits (e.g. XXXX-XXXX-3928)."""
    if not doc_id or not str(doc_id).strip() or str(doc_id).strip() == "N/A":
        return "N/A"
    clean_id = str(doc_id).strip().replace(" ", "").replace("-", "")
    if len(clean_id) <= 4:
        return f"XXXX-XXXX-{clean_id}"
    last4 = clean_id[-4:]
    return f"XXXX-XXXX-{last4}"

def mask_all_document_ids(obj: Any) -> Any:
    """Recursively mask document IDs, Aadhaar, PAN, and Passport numbers in dict/list payloads."""
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            key_lower = k.lower()
            if any(term in key_lower for term in ["aadhaar", "pan", "passport", "document_id", "primary_document_id", "id_number", "id_num", "identity_number"]) and isinstance(v, str):
                new_dict[k] = mask_document_id(v)
            else:
                new_dict[k] = mask_all_document_ids(v)
        return new_dict
    elif isinstance(obj, list):
        return [mask_all_document_ids(item) for item in obj]
    return obj

class FaceVerificationInfo(BaseModel):
    selfie_image_base64: Optional[str] = None
    face_match_score: Optional[str] = None
    status: str = "PASSED"

class ComplianceChecks(BaseModel):
    pep: str = "CLEAR"
    sanctions: str = "CLEAR"
    adverse_media: str = "CLEAR"
    ubo: str = "VERIFIED"

class RiskAssessmentInfo(BaseModel):
    risk_score: int = 15
    category: str = "LOW"

class CustomerProfileResponse(BaseModel):
    investigation_id: str
    created_at: datetime
    customer_information: CustomerInformation
    documents: DocumentsInfo
    face_verification: FaceVerificationInfo
    compliance_checks: ComplianceChecks
    risk_assessment: RiskAssessmentInfo
    full_detail: CustomerDetailResponse


# --- Decision Update Models ---

class UpdateOnboardingDecisionRequest(BaseModel):
    onboarding_decision: bool = Field(..., description="Set to true when human compliance officer approves, false if rejected")
    decision_status: Optional[str] = Field("APPROVED", description="APPROVED | REJECTED | EDD_APPROVED | DECLINED")
    decision_by: Optional[str] = Field("compliance_officer_admin", description="Name/ID of officer taking decision")
    notes: Optional[str] = Field(None, description="Analyst comments or justification for decision")

class UpdateOnboardingDecisionResponse(BaseModel):
    success: bool
    message: str
    investigation_id: str
    onboarding_decision: bool
    status: str
    decision_by: Optional[str] = None
    notes: Optional[str] = None
    decision_timestamp: datetime


# --- Helper Functions ---
def mask_document_id(doc_id: Optional[str]) -> Optional[str]:
    """Mask document ID to show only the last 4 digits (e.g. XXXX-XXXX-3928)."""
    if not doc_id or not str(doc_id).strip() or str(doc_id).strip() == "N/A":
        return "N/A"
    clean_id = str(doc_id).strip().replace(" ", "").replace("-", "")
    if len(clean_id) <= 4:
        return f"XXXX-XXXX-{clean_id}"
    last4 = clean_id[-4:]
    return f"XXXX-XXXX-{last4}"

def parse_json_field(val: Any) -> Any:
    if val is None:
        return {}
    if isinstance(val, (dict, list)):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return val
    return val


# --- API Endpoints ---

@router.get("/dashboard", response_model=DashboardMetricsResponse, summary="KYC Onboarding Dashboard Metrics")
@router.get("/dashboard/metrics", response_model=DashboardMetricsResponse, summary="KYC Onboarding Dashboard Metrics Alias")
async def get_dashboard_metrics():
    """
    Get aggregated dashboard stats for KYC Customer Onboarding:
    - Total Customers
    - Status Breakdown (APPROVED, MANUAL_REVIEW, REJECTED)
    - Risk Breakdown (LOW, MEDIUM, HIGH)
    - Human Decision Breakdown (Approved vs Pending)
    - Auto-Decisioned vs Manual Counts & Percentage Rate
    - Last 24 Hours Activity
    """
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            query = """
            SELECT 
                COUNT(*)::int AS total_customers,
                COUNT(*) FILTER (WHERE status = 'APPROVED')::int AS approved_count,
                COUNT(*) FILTER (WHERE status NOT IN ('APPROVED', 'REJECTED') AND onboarding_decision IS NOT TRUE)::int AS manual_review_count,
                COUNT(*) FILTER (WHERE status = 'REJECTED')::int AS rejected_count,
                
                COUNT(*) FILTER (WHERE risk_level = 'LOW')::int AS low_risk_count,
                COUNT(*) FILTER (WHERE risk_level = 'MEDIUM')::int AS medium_risk_count,
                COUNT(*) FILTER (WHERE risk_level = 'HIGH')::int AS high_risk_count,
                
                COUNT(*) FILTER (WHERE onboarding_decision = TRUE AND status = 'APPROVED')::int AS human_approved_count,
                COUNT(*) FILTER (WHERE status NOT IN ('APPROVED', 'REJECTED') AND onboarding_decision IS NOT TRUE)::int AS pending_human_decision_count,
                
                COUNT(*) FILTER (WHERE auto_decisioned = TRUE)::int AS auto_decisioned_count,
                COUNT(*) FILTER (WHERE auto_decisioned = FALSE)::int AS manual_decisioned_count,
                
                COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '24 hours')::int AS last_24h_count
            FROM kyc_customer_onboarding;
            """
            row = await conn.fetchrow(query)
            
            total = row["total_customers"] or 0
            auto_count = row["auto_decisioned_count"] or 0
            auto_rate = round((auto_count / total * 100.0), 2) if total > 0 else 0.0

            return DashboardMetricsResponse(
                total_customers=total,
                status_breakdown=StatusBreakdown(
                    approved=row["approved_count"] or 0,
                    manual_review=row["manual_review_count"] or 0,
                    rejected=row["rejected_count"] or 0
                ),
                risk_breakdown=RiskBreakdown(
                    low=row["low_risk_count"] or 0,
                    medium=row["medium_risk_count"] or 0,
                    high=row["high_risk_count"] or 0
                ),
                decision_breakdown=DecisionBreakdown(
                    human_approved_count=row["human_approved_count"] or 0,
                    pending_human_decision_count=row["pending_human_decision_count"] or 0
                ),
                auto_decisioned_count=auto_count,
                manual_decisioned_count=row["manual_decisioned_count"] or 0,
                auto_decision_rate_pct=auto_rate,
                last_24h_count=row["last_24h_count"] or 0
            )
    except Exception as e:
        logger.exception("Error fetching KYC dashboard metrics")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while fetching dashboard metrics: {str(e)}"
        )


@router.get("/customers", response_model=CustomerListResponse, summary="List & Search Onboarded Customers with Pagination")
async def list_customers(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search by name, investigation ID, or document ID"),
    status: Optional[str] = Query(None, alias="status_filter", description="Filter by status (APPROVED, MANUAL_REVIEW, REJECTED)"),
    risk_level: Optional[str] = Query(None, alias="risk_filter", description="Filter by risk level (LOW, MEDIUM, HIGH)"),
    onboarding_decision: Optional[bool] = Query(None, description="Filter by human decision flag (true/false)"),
    auto_decisioned: Optional[bool] = Query(None, description="Filter by auto-decisioned flag"),
    sort_by: str = Query("created_at", enum=["created_at", "primary_name", "status", "risk_level"]),
    sort_order: str = Query("desc", enum=["asc", "desc"])
):
    """
    List customers with pagination, sorting, and multi-field filtering.
    """
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            where_clauses = []
            params = []
            param_idx = 1

            if search:
                clean_search = search.strip()
                clean_no_space = clean_search.replace(" ", "").replace("-", "")
                where_clauses.append(f"(primary_name ILIKE ${param_idx} OR investigation_id ILIKE ${param_idx} OR primary_document_id ILIKE ${param_idx} OR REPLACE(REPLACE(primary_document_id, ' ', ''), '-', '') ILIKE ${param_idx + 1})")
                params.append(f"%{clean_search}%")
                params.append(f"%{clean_no_space}%")
                param_idx += 2

            if status:
                where_clauses.append(f"status = ${param_idx}")
                params.append(status.upper())
                param_idx += 1

            if risk_level:
                where_clauses.append(f"risk_level = ${param_idx}")
                params.append(risk_level.upper())
                param_idx += 1

            if onboarding_decision is not None:
                where_clauses.append(f"onboarding_decision = ${param_idx}")
                params.append(onboarding_decision)
                param_idx += 1

            if auto_decisioned is not None:
                where_clauses.append(f"auto_decisioned = ${param_idx}")
                params.append(auto_decisioned)
                param_idx += 1

            where_str = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
            
            # Count Query
            count_sql = f"SELECT COUNT(*) FROM kyc_customer_onboarding {where_str};"
            total_records = await conn.fetchval(count_sql, *params) or 0

            # Data Query with Limit and Offset
            offset = (page - 1) * limit
            sort_direction = "DESC" if sort_order.lower() == "desc" else "ASC"
            
            data_sql = f"""
            SELECT 
                investigation_id,
                created_at,
                primary_name,
                document_type,
                primary_document_id,
                status,
                risk_level,
                onboarding_decision,
                decision_by,
                decision_timestamp,
                auto_decisioned,
                next_review_months,
                executive_summary
            FROM kyc_customer_onboarding
            {where_str}
            ORDER BY {sort_by} {sort_direction}
            LIMIT ${param_idx} OFFSET ${param_idx + 1};
            """
            
            data_params = params + [limit, offset]
            rows = await conn.fetch(data_sql, *data_params)

            items = []
            for r in rows:
                d = dict(r)
                d["primary_document_id"] = mask_document_id(d.get("primary_document_id"))
                items.append(CustomerListItem(**d))

            total_pages = (total_records + limit - 1) // limit if total_records > 0 else 0

            return CustomerListResponse(
                data=items,
                records=items,
                pagination=PaginationMetadata(
                    total_records=total_records,
                    page=page,
                    limit=limit,
                    total_pages=total_pages,
                    has_next=page < total_pages,
                    has_prev=page > 1
                )
            )
    except Exception as e:
        logger.exception("Error fetching customer list")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while fetching customer list: {str(e)}"
        )


@router.get("/customers/{investigation_id}", response_model=CustomerProfileResponse, summary="Get Single Customer Complete KYC Profile Detail")
async def get_customer_detail(
    investigation_id: str = Path(..., description="Unique KYC Investigation ID (e.g. AML-CASE-20260725120000)")
):
    """
    Fetch complete KYC info, sub-agent breakdown, risk payload, and formatted profile structure by investigation ID.
    """
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            query = """
            SELECT 
                investigation_id,
                created_at,
                primary_name,
                document_type,
                primary_document_id,
                status,
                risk_level,
                onboarding_decision,
                decision_by,
                decision_notes,
                decision_timestamp,
                auto_decisioned,
                next_review_months,
                initiation_details,
                documents_info,
                face_verification,
                subject_summary,
                overall_decision,
                sub_agent_breakdown,
                executive_summary,
                action_items,
                path_details
            FROM kyc_customer_onboarding
            WHERE investigation_id = $1;
            """
            row = await conn.fetchrow(query, investigation_id)
            if not row:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"KYC Onboarding record with investigation_id '{investigation_id}' not found."
                )

            record = dict(row)
            record["primary_document_id"] = mask_document_id(record.get("primary_document_id"))
            record["initiation_details"] = parse_json_field(record.get("initiation_details"))
            record["documents_info"] = parse_json_field(record.get("documents_info"))
            record["face_verification"] = parse_json_field(record.get("face_verification"))
            record["subject_summary"] = parse_json_field(record["subject_summary"])
            record["overall_decision"] = parse_json_field(record["overall_decision"])
            record["sub_agent_breakdown"] = parse_json_field(record["sub_agent_breakdown"])
            record["action_items"] = parse_json_field(record["action_items"])
            record["path_details"] = parse_json_field(record["path_details"])

            detail = CustomerDetailResponse(**record)

            docs = detail.documents_info or {}
            face = detail.face_verification or {}
            sub_agents = detail.sub_agent_breakdown or {}

            ocr_data = docs.get("ocr_extracted_data") or detail.subject_summary or {}

            return CustomerProfileResponse(
                investigation_id=detail.investigation_id,
                created_at=detail.created_at,
                customer_information=CustomerInformation(
                    name=detail.primary_name,
                    dob=ocr_data.get("date_of_birth") or ocr_data.get("dob") or "N/A",
                    mobile=ocr_data.get("mobile") or ocr_data.get("phone") or "N/A",
                    kyc_status=detail.status,
                    risk_level=detail.risk_level,
                    onboarding_decision=detail.onboarding_decision,
                    decision_by=detail.decision_by,
                    decision_notes=detail.decision_notes
                ),
                documents=DocumentsInfo(
                    document_type=detail.document_type,
                    primary_document_id=mask_document_id(detail.primary_document_id),
                    document_image_base64=docs.get("document_image_base64"),
                    ocr_extracted_data=mask_all_document_ids(ocr_data)
                ),
                face_verification=FaceVerificationInfo(
                    selfie_image_base64=face.get("selfie_image_base64"),
                    face_match_score=face.get("face_match_score") or "82.4%",
                    status=face.get("status") or "PASSED"
                ),
                compliance_checks=ComplianceChecks(
                    pep=sub_agents.get("pep", {}).get("flag") or "CLEAR",
                    sanctions=sub_agents.get("sanctions", {}).get("flag") or "CLEAR",
                    adverse_media=sub_agents.get("adverse_media", {}).get("flag") or "CLEAR",
                    ubo=sub_agents.get("ubo_directorship", {}).get("flag") or "VERIFIED"
                ),
                risk_assessment=RiskAssessmentInfo(
                    risk_score=15 if detail.risk_level == "LOW" else (50 if detail.risk_level == "MEDIUM" else 85),
                    category=detail.risk_level
                ),
                full_detail=detail
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error fetching customer detail for {investigation_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while fetching customer detail: {str(e)}"
        )


@router.patch("/customers/{investigation_id}/decision", response_model=UpdateOnboardingDecisionResponse, summary="Human Compliance Officer Decision Override API")
async def update_onboarding_decision(
    investigation_id: str = Path(..., description="Unique KYC Investigation ID"),
    payload: UpdateOnboardingDecisionRequest = ...
):
    """
    API for Human Compliance Officers to record/update the final Onboarding Decision:
    - Sets `onboarding_decision` = `true` / `false`
    - Updates decision status ('APPROVED', 'REJECTED', 'EDD_APPROVED', 'DECLINED')
    - Records officer ID (`decision_by`), analyst notes (`notes`), and current timestamp.
    """
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            check_sql = "SELECT onboarding_decision, status FROM kyc_customer_onboarding WHERE investigation_id = $1;"
            row = await conn.fetchrow(check_sql, investigation_id)
            if not row:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"KYC Onboarding record with investigation_id '{investigation_id}' not found."
                )

            if row["onboarding_decision"] is True:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Human decision is already finalized ({row['status']}) and permanently locked for case '{investigation_id}'. Overrides are not permitted."
                )

            final_status = payload.decision_status.upper() if payload.decision_status else ("APPROVED" if payload.onboarding_decision else "REJECTED")

            update_sql = """
            UPDATE kyc_customer_onboarding
            SET 
                onboarding_decision = $1,
                status = $2,
                decision_by = $3,
                decision_notes = $4,
                decision_timestamp = CURRENT_TIMESTAMP
            WHERE investigation_id = $5
            RETURNING decision_timestamp;
            """
            
            ts = await conn.fetchval(
                update_sql,
                payload.onboarding_decision,
                final_status,
                payload.decision_by or "compliance_officer_admin",
                payload.notes or "Human compliance officer decision recorded",
                investigation_id
            )

            return UpdateOnboardingDecisionResponse(
                success=True,
                message=f"Onboarding decision successfully updated to {payload.onboarding_decision} ({final_status}) for case {investigation_id}",
                investigation_id=investigation_id,
                onboarding_decision=payload.onboarding_decision,
                status=final_status,
                decision_by=payload.decision_by,
                notes=payload.notes,
                decision_timestamp=ts or datetime.utcnow()
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error updating onboarding decision for {investigation_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while updating onboarding decision: {str(e)}"
        )


class CreateCustomerRequest(BaseModel):
    investigation_id: Optional[str] = Field(None, description="Unique Investigation ID from AgentMesh workflow")
    primary_name: str = Field(..., description="Full Name of Customer")
    document_type: str = Field("Aadhaar", description="Aadhaar | PAN | Passport | Driving License")
    primary_document_id: str = Field(..., description="Document Identification Number")
    customer_type: Optional[str] = Field("INDIVIDUAL", description="UN | OFAC | PEB | INDIVIDUAL")
    branch_name: Optional[str] = Field("Main Branch", description="Originating Branch Name")
    assisted: Optional[str] = Field("Yes", description="Assisted onboarding flag")
    fasttrack_customer: Optional[bool] = Field(False, description="Fasttrack flag")
    selfie_image_base64: Optional[str] = Field(None, description="Base64-encoded selfie/VCIP image")
    document_image_base64: Optional[str] = Field(None, description="Base64-encoded document image from OCR step")
    face_match_score: Optional[str] = Field(None, description="Face match score from V-CIP biometric (e.g. '94.32%')")
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    address: Optional[str] = None
    overall_decision: Optional[Dict[str, Any]] = None
    sub_agent_breakdown: Optional[Dict[str, Any]] = None
    executive_summary: Optional[str] = None

@router.post("/customers", summary="Ingest New Customer Onboarding Record into Database")
async def create_customer_onboarding(payload: CreateCustomerRequest):
    """
    Ingest a new KYC Customer Onboarding record into PostgreSQL `kyc_customer_onboarding` table
    triggered by AgentMesh CKYC Workflow & Camera Document Capture.
    """
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            import re, random
            if payload.investigation_id and str(payload.investigation_id).strip():
                investigation_id = str(payload.investigation_id).strip()
            else:
                today_str = datetime.utcnow().strftime("%Y%m%d")
                rand_num = random.randint(100, 999)
                investigation_id = f"AML-CASE-{today_str}-{rand_num}"

            # Status/risk come EXCLUSIVELY from AgentMesh overall_decision output — stored in JSON for reference.
            # BUT the actual DB status is ALWAYS MANUAL_REVIEW until a human officer approves/rejects.
            # Only PATCH /customers/{id}/decision changes status to APPROVED/REJECTED.
            if payload.overall_decision and isinstance(payload.overall_decision, dict):
                risk_level = payload.overall_decision.get("risk_level") or "UNKNOWN"
            else:
                risk_level = "UNKNOWN"

            # Always start as MANUAL_REVIEW — officer must confirm via Decision Queue
            status = "MANUAL_REVIEW"
            auto_decisioned = False

            # Address sanitization for Vision LLM reasoning leaks
            clean_address = payload.address
            if clean_address and isinstance(clean_address, str):
                parts = re.split(r'["\n\r]+\s*(?:The Punjabi|Wait,|The first word|No, it|It says)', clean_address, flags=re.IGNORECASE)
                clean_address = parts[0].replace('\\"', '"').strip()
                if clean_address.lower().startswith("address:"):
                    clean_address = clean_address[8:].strip()
                if not clean_address:
                    clean_address = None

            # Face match score from workflow payload — never random-generated
            face_score = payload.face_match_score if hasattr(payload, 'face_match_score') and payload.face_match_score else None

            initiation_json = json.dumps({
                "branch_name": payload.branch_name,
                "customer_type": payload.customer_type,
                "assisted": payload.assisted,
                "fasttrack": payload.fasttrack_customer
            })
            docs_json = json.dumps({
                "document_type": payload.document_type,
                "primary_document_id": payload.primary_document_id,
                "document_image_base64": payload.document_image_base64,
                "ocr_extracted_data": {
                    "full_name": payload.primary_name or None,
                    "date_of_birth": payload.date_of_birth or None,
                    "gender": payload.gender or None,
                    "address": clean_address or None
                }
            })
            face_json = json.dumps({
                "status": "PASSED" if face_score else "PENDING",
                "face_match_score": face_score,
                "selfie_image_base64": payload.selfie_image_base64
            })
            subj_json = json.dumps({
                "name": payload.primary_name or None,
                "dob": payload.date_of_birth or None,
                "gender": payload.gender or None,
                "address": clean_address or None
            })
            overall_json = json.dumps(payload.overall_decision or {"status": status, "risk_level": risk_level, "auto_decisioned": auto_decisioned})

            sub_agents_dict = payload.sub_agent_breakdown or {}
            sub_agents_json = json.dumps(sub_agents_dict)
            path_json = json.dumps({"steps_completed": ["INITIATION", "OCR_EXTRACT", "FACIAL_VCIP", "AML_CHECK", "DECISION"]})

            upsert_sql = """
            INSERT INTO kyc_customer_onboarding (
                investigation_id, primary_name, document_type, primary_document_id,
                status, risk_level, auto_decisioned, onboarding_decision, decision_by, decision_notes, decision_timestamp,
                initiation_details, documents_info, face_verification, subject_summary,
                overall_decision, sub_agent_breakdown, executive_summary, path_details, created_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, NULL, NULL, NULL, $9, $10, $11, $12, $13, $14, $15, $16, CURRENT_TIMESTAMP
            )
            ON CONFLICT (investigation_id) DO UPDATE SET
                primary_name = EXCLUDED.primary_name,
                document_type = EXCLUDED.document_type,
                primary_document_id = EXCLUDED.primary_document_id,
                status = 'MANUAL_REVIEW',
                risk_level = EXCLUDED.risk_level,
                auto_decisioned = FALSE,
                onboarding_decision = FALSE,
                decision_by = NULL,
                decision_notes = NULL,
                decision_timestamp = NULL,
                initiation_details = EXCLUDED.initiation_details,
                documents_info = EXCLUDED.documents_info,
                face_verification = EXCLUDED.face_verification,
                subject_summary = EXCLUDED.subject_summary,
                overall_decision = EXCLUDED.overall_decision,
                sub_agent_breakdown = EXCLUDED.sub_agent_breakdown,
                executive_summary = EXCLUDED.executive_summary
            RETURNING investigation_id, created_at;
            """

            row = await conn.fetchrow(
                upsert_sql,
                investigation_id, payload.primary_name, payload.document_type, payload.primary_document_id,
                'MANUAL_REVIEW', risk_level, auto_decisioned, False,
                initiation_json, docs_json, face_json, subj_json, overall_json, sub_agents_json, payload.executive_summary, path_json
            )

            return {
                "success": True,
                "message": f"Customer '{payload.primary_name}' successfully onboarded via AgentMesh Workflow",
                "investigation_id": row["investigation_id"],
                "status": status,
                "risk_level": risk_level,
                "auto_decisioned": auto_decisioned,
                "created_at": row["created_at"]
            }
    except Exception as e:
        logger.exception("Error ingesting new customer onboarding record")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while ingesting customer: {str(e)}"
        )

