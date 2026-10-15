"""
Customer 360 API — B2B Fraud Detection
India-localized, 145 fields, deterministic per-account matching.

Run: uvicorn customer360_api:app --host 0.0.0.0 --port 8001

Matching mechanism:
  Each customer profile is generated with random.seed(customer_index),
  so ACC100001 always returns the same 145-field profile on every restart.
  Lookup dicts are keyed by both account_id AND customer_id for flexible access.
"""

import random
from datetime import datetime, timedelta, timezone
from faker import Faker

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# ── Faker seeded for reproducibility (locale en_IN for Indian data) ──────────
fake_master = Faker("en_IN")
Faker.seed(42)

# ── Indian geography ─────────────────────────────────────────────────────────
INDIAN_CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Surat",
    "Lucknow", "Kanpur", "Nagpur", "Indore", "Bhopal",
    "Coimbatore", "Kochi", "Chandigarh", "Vadodara", "Nashik",
]
INDIAN_STATES = [
    "Maharashtra", "Karnataka", "Tamil Nadu", "Telangana", "Gujarat",
    "Rajasthan", "Uttar Pradesh", "West Bengal", "Delhi", "Kerala",
    "Madhya Pradesh", "Punjab", "Haryana", "Bihar", "Andhra Pradesh",
]
# Rough bounding box for India
INDIA_LAT = (8.4, 37.6)
INDIA_LON = (68.7, 97.4)

# ── NIC 2008 codes relevant to Indian B2B ────────────────────────────────────
NIC_CODES = {
    "4510": "Sale of motor vehicles",
    "4610": "Agents involved in wholesale trade",
    "4620": "Wholesale of agricultural raw materials",
    "4630": "Wholesale of food, beverages and tobacco",
    "4641": "Wholesale of textiles",
    "4649": "Wholesale of other household goods",
    "4661": "Wholesale of solid, liquid and gaseous fuels",
    "4669": "Wholesale of other machinery and equipment",
    "4711": "Retail sale of food, beverages and tobacco",
    "6110": "Wired telecommunications activities",
    "6201": "Computer programming activities",
    "6209": "Other information technology services",
    "6411": "Central banking",
    "6419": "Other monetary intermediation",
    "6499": "Other financial service activities",
    "6810": "Real estate activities with own or leased property",
    "7010": "Activities of head offices",
    "7020": "Management consultancy activities",
    "7210": "Research and experimental development on natural sciences",
    "8010": "Private security activities",
    "8020": "Security systems service activities",
    "1010": "Processing and preserving of meat",
    "1410": "Manufacture of wearing apparel",
    "2410": "Manufacture of basic iron and steel",
    "2610": "Manufacture of electronic components",
    "2710": "Manufacture of electric motors and generators",
    "3110": "Manufacture of furniture",
    "5210": "Warehousing and storage",
    "5320": "Other postal and courier activities",
}

# ── Lookup tables ─────────────────────────────────────────────────────────────
# Payment corridors — mostly India-centric with some cross-border
TXN_CORRIDORS = ["IN", "AE", "US", "SG", "GB", "DE", "JP", "MY", "HK", "QA"]

PAYMENT_RAILS = ["NEFT", "RTGS", "UPI", "IMPS", "SWIFT", "NACH"]

ACCOUNT_TYPES = ["current", "cash_credit", "overdraft", "escrow", "trade_finance", "savings"]

PRODUCTS = [
    "current_account", "cash_management", "trade_finance",
    "fx_forward", "letter_of_credit", "bank_guarantee",
    "term_loan", "working_capital_loan", "credit_card",
    "insurance", "mutual_funds", "treasury_services", "payroll",
]

DATA_SOURCES = [
    "core_banking", "kyc_vendor", "transaction_monitoring",
    "credit_bureau", "crm", "card_management_system", "complaints_system",
]

MERCHANT_CATEGORIES = [
    "raw_materials", "wholesale_trade", "manufacturing_inputs",
    "logistics", "utilities", "professional_services",
    "it_services", "equipment_purchase", "travel_corporate",
]

ENTITY_TYPES = ["private_limited", "public_limited", "llp", "partnership", "proprietorship", "trust"]
SEGMENTS     = ["sme", "mid_corporate", "large_corporate", "institutional", "startup"]
REL_TIERS    = ["standard", "priority", "institutional", "ultra_hni"]
KYC_TIERS    = ["low", "medium", "high"]
AML_RATINGS  = ["low", "medium", "high"]
CDD_LEVELS   = ["standard", "enhanced", "simplified"]
SEASONALITY  = ["flat", "quarter_end_spike", "month_end_spike", "festive_spike"]
CREDIT_RTGS  = ["AAA", "AA", "A", "BBB", "BB", "B", "Unrated"]
BUREAU_NAMES = ["CIBIL", "Experian India", "Equifax India", "CRIF High Mark"]

# ── Helper ────────────────────────────────────────────────────────────────────
def _date_str(dt): return dt.strftime("%Y-%m-%d")
def _dt_str(dt):   return dt.isoformat() + "Z"
def _now():        return datetime.now(timezone.utc)

def _past_date(rng, min_days, max_days):
    return _now() - timedelta(days=rng.randint(min_days, max_days))

def _future_date(rng, min_days, max_days):
    return _now() + timedelta(days=rng.randint(min_days, max_days))

def _indian_gstin(rng, pan):
    state_code = str(rng.randint(1, 36)).zfill(2)
    suffix = ''.join(rng.choices("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=9))
    return f"{state_code}{pan[:10]}{suffix[:3]}Z{rng.choice('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')}"

def _indian_phone(rng):
    prefixes = ["98", "97", "96", "95", "94", "93", "92", "91", "90", "89", "88", "87", "86", "85", "84", "83", "82", "81", "80", "79", "78", "77", "76", "75", "74", "73", "72", "71", "70"]
    return "+91-" + rng.choice(prefixes) + ''.join([str(rng.randint(0,9)) for _ in range(8)])

def _indian_company_name(rng, fake):
    suffixes = [" Pvt Ltd", " Ltd", " LLP", " Enterprises", " Industries", " Trading Co", " Solutions", " Group", " Holdings", " Exports"]
    words = [fake.last_name() for _ in range(rng.randint(1,3))]
    return " ".join(words) + rng.choice(suffixes)

# ── Core profile generator ────────────────────────────────────────────────────
def generate_customer(index: int) -> dict:
    """
    Generate a deterministic Customer 360 profile for a given index.
    random.seed(index) ensures the same index always produces the same profile.
    """
    rng = random.Random(index)          # per-customer deterministic RNG
    fake = Faker("en_IN")
    fake.seed_instance(index)

    cid = f"CUST{index:05d}"
    aid = f"ACC10{index:04d}"

    # ── 1. Entity Identity ──────────────────────────────────────────────────
    nic_code      = rng.choice(list(NIC_CODES.keys()))
    industry      = NIC_CODES[nic_code]
    inc_date      = _past_date(rng, 365*3, 365*20)
    entity_type   = rng.choice(ENTITY_TYPES)
    pan           = ''.join(rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5)) + \
                    ''.join(rng.choices("0123456789", k=4)) + \
                    rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    company_name  = _indian_company_name(rng, fake)
    lei           = ''.join(rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=20))

    # tax residency — 90% India-only, 5% dual, 5% other
    tr_roll = rng.random()
    if tr_roll < 0.90:
        tax_residency = ["IN"]
    elif tr_roll < 0.95:
        tax_residency = ["IN", rng.choice(["AE", "SG", "US", "GB"])]
    else:
        tax_residency = [rng.choice(["AE", "SG", "US", "GB", "NL"])]

    # ── 2. Contact & Address ────────────────────────────────────────────────
    city   = rng.choice(INDIAN_CITIES)
    state  = rng.choice(INDIAN_STATES)
    pincode = str(rng.randint(100000, 999999))
    reg_addr  = f"{rng.randint(1,999)}, {fake.street_name()}, {city}, {state} - {pincode}"
    comm_addr = reg_addr if rng.random() < 0.6 else \
                f"{rng.randint(1,999)}, {fake.street_name()}, {rng.choice(INDIAN_CITIES)}, {rng.choice(INDIAN_STATES)} - {str(rng.randint(100000,999999))}"
    geo_lat = round(rng.uniform(*INDIA_LAT), 6)
    geo_lon = round(rng.uniform(*INDIA_LON), 6)

    # ── 3. KYC / AML / Compliance ───────────────────────────────────────────
    kyc_tier        = rng.choices(KYC_TIERS, weights=[60, 30, 10])[0]
    kyc_date        = _past_date(rng, 30, 365*3)
    last_kyc_review = _past_date(rng, 30, 730)
    next_kyc_due    = _future_date(rng, 90, 730)
    pep_flag        = rng.random() < 0.05
    sanctions_score = rng.randint(0, 10) if kyc_tier == "low" else rng.randint(5, 40)
    ubos = [fake.name() for _ in range(rng.randint(1, 4))]
    aml_case_count  = 0 if kyc_tier == "low" else rng.randint(0, 3)
    cdd_level       = {"low": "standard", "medium": "standard", "high": "enhanced"}[kyc_tier]

    # ── 4. Account & Relationship ───────────────────────────────────────────
    acct_open        = inc_date + timedelta(days=rng.randint(1, 90))
    acct_age_days    = (_now() - acct_open).days
    rel_tier         = rng.choices(REL_TIERS, weights=[50, 30, 15, 5])[0]
    segment          = rng.choice(SEGMENTS)
    rm_name          = fake.name()
    rm_branch        = rng.choice(INDIAN_CITIES)
    n_accounts       = rng.randint(1, 8)
    all_accounts     = [aid] + [f"ACC2{index:04d}{i}" for i in range(1, n_accounts)]
    products_used    = rng.sample(PRODUCTS, k=rng.randint(2, 6))
    account_type     = rng.choice(ACCOUNT_TYPES)

    # ── 5. Credit & Risk ───────────────────────────────────────────────────
    internal_score   = rng.randint(300, 900)
    bureau_score     = rng.randint(300, 900)
    credit_rating    = rng.choice(CREDIT_RTGS)
    credit_limit     = round(rng.uniform(500_000, 50_000_000), 2)
    exposure         = round(rng.uniform(0, credit_limit * 0.8), 2)
    utilization      = round(exposure / credit_limit, 3)
    dpd              = rng.choices([0, 30, 60, 90], weights=[75, 12, 8, 5])[0]
    npa_flag         = dpd >= 90
    pd_score         = round(rng.uniform(0.001, 0.15), 4)
    health_score     = round(rng.uniform(40, 99), 1)
    collateral       = round(rng.uniform(0, credit_limit * 1.2), 2)
    ltv              = round(exposure / collateral, 3) if collateral > 0 else None

    # ── 6. Transaction Behavior ─────────────────────────────────────────────
    avg_amt_12m   = round(rng.uniform(5_000, 500_000), 2)
    avg_amt_90d   = round(avg_amt_12m * rng.uniform(0.8, 1.3), 2)
    avg_amt_30d   = round(avg_amt_12m * rng.uniform(0.7, 1.4), 2)
    avg_amt_7d    = round(avg_amt_12m * rng.uniform(0.6, 1.5), 2)
    max_amt_90d   = round(avg_amt_90d * rng.uniform(1.5, 4.0), 2)
    monthly_vol   = rng.randint(10, 400)
    # corridors — mostly IN, may have 1-2 foreign
    n_corridors   = rng.randint(1, 4)
    corridors     = ["IN"] + rng.sample([c for c in TXN_CORRIDORS if c != "IN"], k=min(n_corridors-1, len(TXN_CORRIDORS)-1))
    rails         = rng.sample(PAYMENT_RAILS, k=rng.randint(2, 5))
    peak_hours    = sorted(rng.sample(range(7, 21), k=3))   # Indian business hours
    merchant_cats = rng.sample(MERCHANT_CATEGORIES, k=rng.randint(2, 4))
    channel_usage = {
        "branch":           round(rng.uniform(5, 60), 1),
        "internet_banking": round(rng.uniform(20, 90), 1),
        "mobile_banking":   round(rng.uniform(5, 50), 1),
        "api_corporate":    round(rng.uniform(0, 80), 1),
    }

    # ── 7. Digital & Channel Engagement ────────────────────────────────────
    n_devices     = rng.randint(1, 8)
    primary_ip    = rng.choices(["IN"] + [c for c in TXN_CORRIDORS if c != "IN"], weights=[85]+[1]*9)[0]
    digital_pct   = round(rng.uniform(10, 95), 1)
    adopt_score   = round(rng.uniform(10, 100), 1)
    biometric     = rng.choice(["enrolled", "partial", "not_enrolled"])
    last_login    = _past_date(rng, 1, 90)
    login_freq_30 = rng.randint(5, 120)
    login_anomaly = rng.choices([0, 1, 2, 3], weights=[80, 12, 5, 3])[0]
    pref_channel  = rng.choice(["internet_banking", "mobile_banking", "branch", "api_corporate"])
    nps           = rng.randint(-100, 100)
    csat          = round(rng.uniform(1, 5), 1)

    # ── 8. Fraud & Dispute History ──────────────────────────────────────────
    fraud_flags   = rng.choices([0, 1, 2], weights=[85, 12, 3])[0]
    confirmed_fraud = 0 if fraud_flags == 0 else rng.randint(0, fraud_flags)
    chargebacks   = rng.choices([0, 1, 2, 3], weights=[70, 18, 8, 4])[0]
    disputes      = rng.choices([0, 1, 2, 3], weights=[65, 20, 10, 5])[0]
    blocked_cards = rng.randint(0, 3)
    watchlist_hits= 0 if kyc_tier == "low" else rng.randint(0, 2)
    adverse_media = rng.random() < 0.03
    sar_filed     = rng.random() < 0.02
    fraud_outcome = "n/a" if confirmed_fraud == 0 else rng.choice(["closed_confirmed", "closed_inconclusive"])

    # ── 9. Servicing & Complaints ───────────────────────────────────────────
    complaint_count  = rng.randint(0, 5)
    resolution_tat   = round(rng.uniform(1, 15), 1) if complaint_count > 0 else None
    service_tickets  = rng.randint(0, 8)
    branch_visits    = rng.randint(0, 24)
    call_center_int  = rng.randint(0, 20)
    nps_trend        = rng.choice(["improving", "stable", "declining"])

    # ── 10. Product Holding & Cross-sell ───────────────────────────────────
    products_held    = rng.sample(PRODUCTS, k=rng.randint(1, 5))
    lapsed_products  = rng.sample(PRODUCTS, k=rng.randint(0, 2))
    propensity_scores = {
        p: round(rng.uniform(0.1, 0.95), 2)
        for p in rng.sample(["mutual_funds", "credit_card", "insurance", "fx_forward", "term_loan"], k=rng.randint(2,4))
    }
    active_offers    = rng.randint(0, 3)
    last_offer_resp  = rng.choice(["accepted", "rejected", "pending", "n/a"])

    # ── 11. Beneficiary Network ─────────────────────────────────────────────
    known_benes      = rng.randint(2, 100)
    new_benes_30d    = rng.randint(0, min(10, known_benes))
    bene_conc_ratio  = round(new_benes_30d / max(known_benes, 1), 3)
    related_entities = [_indian_company_name(rng, fake) for _ in range(rng.randint(0, 3))]
    trade_cps        = [_indian_company_name(rng, fake) for _ in range(rng.randint(0, 2))]
    guarantors       = [fake.name() for _ in range(rng.randint(0, 2))]
    corridor_hr      = rng.random() < 0.05   # 5% have high-risk corridor exposure

    # ── 12. Regulatory / Reporting ──────────────────────────────────────────
    fatca_status     = rng.choices(["not_reportable", "reportable", "pending"], weights=[88, 5, 7])[0]
    crs_status       = rng.choices(["not_reportable", "reportable", "pending"], weights=[88, 5, 7])[0]
    large_txn_flag   = max_amt_90d > 1_000_000
    reg_filings      = rng.choices(["current", "overdue", "n/a"], weights=[80, 10, 10])[0]
    gst_filing       = rng.choices(["current", "overdue", "n/a"], weights=[70, 15, 15])[0]
    last_audit       = _past_date(rng, 90, 730)
    aml_risk         = rng.choices(AML_RATINGS, weights=[70, 22, 8])[0]
    reg_cap_class    = rng.choices(
        ["standard", "watch", "substandard", "doubtful", "loss"],
        weights=[70, 15, 8, 5, 2]
    )[0]
    fatca_crs        = rng.choices(["compliant", "pending", "exempt"], weights=[80, 12, 8])[0]

    # ── 13. Metadata ────────────────────────────────────────────────────────
    sources          = rng.sample(DATA_SOURCES, k=rng.randint(4, 7))
    dq_score         = round(rng.uniform(60, 99), 1)
    gr_version       = f"v{rng.randint(1,9)}.{rng.randint(0,9)}"
    profile_updated  = _dt_str(_past_date(rng, 0, 7))

    return {
        # ── 1. Entity Identity ──────────────────────────────────────────
        "customer_id":                      cid,
        "account_id":                       aid,
        "legal_entity_name":                company_name,
        "dba_name":                         company_name,
        "entity_type":                      entity_type,
        "incorporation_date":               _date_str(inc_date),
        "incorporation_country":            "IN",
        "nationality_of_promoters":         "IN",
        "business_registration_number":     f"REG-{rng.randint(10000000,99999999)}",
        "tax_id":                           pan,
        "gstin":                            _indian_gstin(rng, pan),
        "lei_code":                         lei,
        "industry_sector":                  industry,
        "nic_code":                         nic_code,
        "company_size_employees":           rng.randint(2, 10000),
        "annual_revenue_inr":               round(rng.uniform(1_000_000, 5_000_000_000), 2),
        "fatca_status":                     fatca_status,
        "crs_status":                       crs_status,
        "tax_residency_countries":          tax_residency,
        # ── 2. Contact & Address ────────────────────────────────────────
        "registered_address":               reg_addr,
        "communication_address":            comm_addr,
        "city":                             city,
        "state":                            state,
        "pincode":                          pincode,
        "address_verified":                 rng.random() < 0.92,
        "address_last_updated":             _date_str(_past_date(rng, 30, 730)),
        "primary_email":                    fake.company_email(),
        "email_verified":                   rng.random() < 0.95,
        "secondary_email":                  fake.company_email(),
        "primary_phone":                    _indian_phone(rng),
        "phone_verified":                   rng.random() < 0.95,
        "secondary_phone":                  _indian_phone(rng),
        "registered_geo_lat":               geo_lat,
        "registered_geo_lon":               geo_lon,
        "mailing_preference":               rng.choice(["email", "sms", "post", "whatsapp"]),
        "address_change_count_12m":         rng.randint(0, 3),
        # ── 3. KYC / AML / Compliance ──────────────────────────────────
        "kyc_status":                       rng.choices(["verified", "pending_review"], weights=[90, 10])[0],
        "kyc_tier":                         kyc_tier,
        "kyc_completion_date":              _date_str(kyc_date),
        "last_kyc_review_date":             _date_str(last_kyc_review),
        "next_kyc_due_date":                _date_str(next_kyc_due),
        "ultimate_beneficial_owners":       ubos,
        "pep_flag":                         pep_flag,
        "pep_relationship_type":            rng.choice(["direct", "associated"]) if pep_flag else "n/a",
        "sanctions_screening_status":       "flagged" if sanctions_score > 35 else "clear",
        "sanctions_match_score":            sanctions_score,
        "ofac_screening_result":            "match" if sanctions_score > 35 else "no_match",
        "adverse_media_screening_result":   "flagged" if adverse_media else "clear",
        "source_of_funds":                  rng.choice(["business_revenue", "trade_proceeds", "investment_returns", "loan_disbursement"]),
        "source_of_wealth_verified":        rng.random() < 0.88,
        "cdd_level":                        cdd_level,
        "edd_required":                     cdd_level == "enhanced",
        "aml_case_history_count":           aml_case_count,
        "fatf_high_risk_country_exposure":  rng.random() < 0.04,
        # ── 4. Account & Relationship ───────────────────────────────────
        "primary_account_id":               aid,
        "all_account_ids":                  all_accounts,
        "account_type":                     account_type,
        "account_open_date":                _date_str(acct_open),
        "account_age_days":                 acct_age_days,
        "relationship_start_date":          _date_str(acct_open),
        "relationship_manager":             rm_name,
        "relationship_manager_branch":      rm_branch,
        "customer_segment":                 segment,
        "banking_relationship_tier":        rel_tier,
        "number_of_accounts":               n_accounts,
        "household_group_id":               f"GRP{rng.randint(10000,99999)}" if rng.random() < 0.4 else None,
        "products_used":                    products_used,
        "cross_sell_product_count":         len(products_used),
        # ── 5. Credit & Risk ────────────────────────────────────────────
        "internal_credit_score":            internal_score,
        "external_bureau_score":            bureau_score,
        "bureau_name":                      rng.choice(BUREAU_NAMES),
        "credit_rating":                    credit_rating,
        "credit_limit_inr":                 credit_limit,
        "outstanding_credit_exposure_inr":  exposure,
        "credit_utilization_ratio":         utilization,
        "days_past_due":                    dpd,
        "npa_flag":                         npa_flag,
        "restructuring_history_flag":       rng.random() < 0.05,
        "collateral_value_inr":             collateral,
        "loan_to_value_ratio":              ltv,
        "probability_of_default_score":     pd_score,
        "financial_health_score":           health_score,
        # ── 6. Transaction Behavior ─────────────────────────────────────
        "avg_txn_amount_7d":                avg_amt_7d,
        "avg_txn_amount_30d":               avg_amt_30d,
        "avg_txn_amount_90d":               avg_amt_90d,
        "avg_txn_amount_12m":               avg_amt_12m,
        "avg_monthly_txn_volume":           monthly_vol,
        "max_txn_amount_90d":               max_amt_90d,
        "channel_usage_pct":                channel_usage,
        "typical_payment_rails":            rails,
        "typical_txn_corridors":            corridors,
        "peak_txn_hours":                   peak_hours,
        "top_merchant_categories":          merchant_cats,
        "recurring_payment_count":          rng.randint(0, 15),
        "dormancy_flag":                    rng.random() < 0.03,
        "txn_velocity_score":               round(rng.uniform(0, 100), 1),
        "seasonality_pattern":              rng.choice(SEASONALITY),
        # ── 7. Digital & Channel Engagement ────────────────────────────
        "registered_devices_count":         n_devices,
        "primary_ip_geolocation":           primary_ip,
        "digital_channel_usage_pct":        digital_pct,
        "digital_adoption_score":           adopt_score,
        "biometric_enrollment_status":      biometric,
        "last_login_timestamp":             _dt_str(last_login),
        "login_frequency_30d":              login_freq_30,
        "login_anomaly_count_90d":          login_anomaly,
        "preferred_channel":                pref_channel,
        "nps_score":                        nps,
        "csat_score":                       csat,
        # ── 8. Fraud & Dispute History ──────────────────────────────────
        "historical_fraud_flags_count":     fraud_flags,
        "confirmed_fraud_count":            confirmed_fraud,
        "chargeback_count_12m":             chargebacks,
        "dispute_count_12m":                disputes,
        "blocked_card_history_count":       blocked_cards,
        "watchlist_hits":                   watchlist_hits,
        "adverse_media_flag":               adverse_media,
        "sar_filed_flag":                   sar_filed,
        "fraud_case_outcomes":              fraud_outcome,
        # ── 9. Servicing & Complaints ───────────────────────────────────
        "complaint_count_12m":              complaint_count,
        "complaint_resolution_avg_tat_days": resolution_tat,
        "service_ticket_count_12m":         service_tickets,
        "branch_visit_count_12m":           branch_visits,
        "call_center_interaction_count_12m": call_center_int,
        "nps_trend":                        nps_trend,
        # ── 10. Product Holding & Cross-sell ────────────────────────────
        "products_held":                    products_held,
        "product_propensity_scores":        propensity_scores,
        "lapsed_products":                  lapsed_products,
        "active_offers_count":              active_offers,
        "last_offer_response":              last_offer_resp,
        # ── 11. Beneficiary Network ─────────────────────────────────────
        "known_beneficiaries_count":        known_benes,
        "new_beneficiaries_last_30d":       new_benes_30d,
        "beneficiary_concentration_ratio":  bene_conc_ratio,
        "related_entities":                 related_entities,
        "trade_counterparties":             trade_cps,
        "guarantor_relationships":          guarantors,
        "corridor_exposure_high_risk":      corridor_hr,
        # ── 12. Regulatory / Reporting ──────────────────────────────────
        "fatca_crs_status":                 fatca_crs,
        "large_txn_reporting_flag":         large_txn_flag,
        "regulatory_filings_status":        reg_filings,
        "gst_filing_status":                gst_filing,
        "last_statutory_audit_date":        _date_str(last_audit),
        "aml_risk_rating":                  aml_risk,
        "regulatory_capital_classification": reg_cap_class,
        # ── 13. Metadata ─────────────────────────────────────────────────
        "profile_last_updated":             profile_updated,
        "data_sources":                     sources,
        "data_quality_score":               dq_score,
        "golden_record_version":            gr_version,
        "consent_marketing_flag":           rng.random() < 0.75,
        "consent_data_sharing_flag":        rng.random() < 0.90,
    }


# ── Build the customer store (100 customers) ──────────────────────────────────
#
# MATCHING MECHANISM:
#   We pre-generate all 100 profiles at startup and index them by BOTH
#   customer_id ("CUST00001") and account_id ("ACC100001").
#   When a transaction arrives carrying account_id = "ACC100050", the api_node
#   calls GET /customer360/by-account/ACC100050 which hits _BY_ACCOUNT dict
#   directly — O(1) lookup, always returns the exact same profile for that
#   account because generate_customer(50) is seeded with index 50.
#
CUSTOMERS: list[dict] = [generate_customer(i) for i in range(1, 101)]
_BY_CUSTOMER_ID: dict[str, dict] = {c["customer_id"]: c for c in CUSTOMERS}
_BY_ACCOUNT_ID:  dict[str, dict] = {c["account_id"]:  c for c in CUSTOMERS}


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer 360 API",
    description="B2B Fraud Detection — India-localized Customer 360 profiles for 100 customers.",
    version="3.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok", "customers_loaded": len(CUSTOMERS)}


@app.get("/customers")
def list_customers():
    """Lightweight list — customer_id, account_id, name, kyc_tier only."""
    return [
        {
            "customer_id": c["customer_id"],
            "account_id":  c["account_id"],
            "legal_entity_name": c["legal_entity_name"],
            "kyc_tier":    c["kyc_tier"],
            "city":        c["city"],
            "segment":     c["customer_segment"],
        }
        for c in CUSTOMERS
    ]


@app.get("/customers/{customer_id}")
def get_customer_by_id(customer_id: str):
    """Full 145-field profile by customer_id (e.g. CUST00050)."""
    profile = _BY_CUSTOMER_ID.get(customer_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"customer_id '{customer_id}' not found")
    return JSONResponse(content=profile)


@app.get("/customer360/by-account/{account_id}")
def get_customer_by_account(account_id: str):
    """
    Full 145-field profile by account_id (e.g. ACC100050).
    This is the endpoint to wire into your api_node — transactions carry account_id,
    so the workflow fetches the exact customer profile for that account.
    """
    profile = _BY_ACCOUNT_ID.get(account_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"account_id '{account_id}' not found")
    return JSONResponse(content=profile)


@app.get("/customer360/accounts")
def list_account_ids():
    """Returns all valid account_ids — useful for testing your workflow with known IDs."""
    return {
        "valid_account_ids": list(_BY_ACCOUNT_ID.keys()),
        "count": len(_BY_ACCOUNT_ID),
    }