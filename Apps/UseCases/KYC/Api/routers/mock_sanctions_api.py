from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from rapidfuzz import fuzz

router = APIRouter(prefix="/mock-sanctions", tags=["Mock Sanctions & OFAC Screening APIs"])

SANCTIONS_DB = [
    {
        "name": "Osama Bin Laden",
        "dob": "1957-03-10",
        "country": "Saudi Arabia",
        "list": "UN"
    },
    {
        "name": "Ayman Al Zawahiri",
        "dob": "1951-06-19",
        "country": "Egypt",
        "list": "UN"
    },
    {
        "name": "Abu Bakr Al Baghdadi",
        "dob": "1971-07-28",
        "country": "Iraq",
        "list": "UN"
    },
    {
        "name": "Mohammed Atef",
        "dob": "1944-01-01",
        "country": "Egypt",
        "list": "OFAC"
    },
    {
        "name": "Saif Al Adel",
        "dob": "1960-04-11",
        "country": "Egypt",
        "list": "OFAC"
    },
    {
        "name": "Sirajuddin Haqqani",
        "dob": "1973-01-01",
        "country": "Afghanistan",
        "list": "UN"
    },
    {
        "name": "Mullah Yaqoob",
        "dob": "1990-01-01",
        "country": "Afghanistan",
        "list": "UN"
    },
    {
        "name": "Abdul Rauf Khadim",
        "dob": "1970-01-01",
        "country": "Afghanistan",
        "list": "OFAC"
    },
    {
        "name": "Ibrahim Aqil",
        "dob": "1962-01-01",
        "country": "Lebanon",
        "list": "OFAC"
    },
    {
        "name": "Hassan Nasrallah",
        "dob": "1960-08-31",
        "country": "Lebanon",
        "list": "OFAC"
    },
    {
        "name": "Abdul Malik Al Houthi",
        "dob": "1979-05-22",
        "country": "Yemen",
        "list": "OFAC"
    },
    {
        "name": "Ismail Haniyeh",
        "dob": "1962-01-29",
        "country": "Palestine",
        "list": "OFAC"
    },
    {
        "name": "Yahya Sinwar",
        "dob": "1962-10-19",
        "country": "Palestine",
        "list": "OFAC"
    },
    {
        "name": "Mohammed Deif",
        "dob": "1965-01-01",
        "country": "Palestine",
        "list": "UN"
    },
    {
        "name": "Ramzan Kadyrov",
        "dob": "1976-10-05",
        "country": "Russia",
        "list": "OFAC"
    },
    {
        "name": "Viktor Bout",
        "dob": "1967-01-13",
        "country": "Russia",
        "list": "UN"
    },
    {
        "name": "Semion Mogilevich",
        "dob": "1946-06-30",
        "country": "Ukraine",
        "list": "OFAC"
    },
    {
        "name": "Joaquin Guzman",
        "dob": "1957-04-04",
        "country": "Mexico",
        "list": "OFAC"
    },
    {
        "name": "Rafael Caro Quintero",
        "dob": "1952-10-24",
        "country": "Mexico",
        "list": "OFAC"
    },
    {
        "name": "Nemesio Oseguera Cervantes",
        "dob": "1966-07-17",
        "country": "Mexico",
        "list": "OFAC"
    }
]

@router.get("/screen")
def screen(name: Optional[str] = None, dob: Optional[str] = None, country: Optional[str] = None):
    search_name = (name or "").strip()
    search_dob = (dob or "").strip()
    search_country = (country or "").strip().lower()

    best_match = None
    best_score = 0.0

    if search_name:
        for person in SANCTIONS_DB:
            # 1. Fuzzy Name Match
            name_score = max(
                fuzz.token_sort_ratio(search_name.lower(), person["name"].lower()),
                fuzz.partial_ratio(search_name.lower(), person["name"].lower())
            )

            current_score = float(name_score)

            # 2. DOB Match / Verification
            dob_matched = False
            if search_dob and person.get("dob"):
                if search_dob == person["dob"] or search_dob in person["dob"] or person["dob"] in search_dob:
                    dob_matched = True

            # 3. Country Match / Verification
            country_matched = False
            if search_country and person.get("country"):
                p_country = person["country"].lower()
                if search_country in p_country or p_country in search_country:
                    country_matched = True

            # 4. Combo Evaluation Logic
            # Combo A: High Name Match + Matching DOB -> Flag as FAIL (98%) even if country is missing/different
            if name_score >= 75 and dob_matched:
                current_score = max(current_score, 98.0)
            # Combo B: High Name Match + Matching Country -> Flag as FAIL (95%) even if DOB is missing/different
            elif name_score >= 80 and country_matched:
                current_score = max(current_score, 95.0)
            # Combo C: Very High Name Match alone (>= 85%) -> Flag as FAIL (90%)
            elif name_score >= 85:
                current_score = max(current_score, 90.0)

            if current_score > best_score:
                best_score = current_score
                best_match = person

    if best_score >= 90:
        status = "FAIL"
    elif best_score >= 75:
        status = "REVIEW"
    else:
        status = "PASS"

    return {
        "status": status,
        "match_score": round(float(best_score), 2),
        "input": {"name": search_name, "dob": search_dob, "country": search_country},
        "matched_record": best_match if status != "PASS" else None
    }
