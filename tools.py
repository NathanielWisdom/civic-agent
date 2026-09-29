"""
tools.py — Tools for the civic engagement agent.

Two tools, kept deliberately dumb and narrow:

1. civic_lookup
   Given an address, returns the contests (races) on that person's ballot,
   via the Google Civic Information API. No LLM involved — a clean fetch.

2. office_responsibilities
   Given an office name + a full jurisdiction, searches the web (biased
   toward official .gov sources) and returns the raw results — titles,
   URLs, snippets — for the agent to read and summarize itself. This tool
   does NOT summarize on its own; keeping that in the main agent loop
   means the summary and the source citation come from the same reasoning
   step, instead of being laundered through an extra layer.

Requires two environment variables:
  GOOGLE_CIVIC_API_KEY  — from console.cloud.google.com (Civic Information API)
  (no key needed for the DuckDuckGo search tool)
"""

import os
import requests
from typing import Optional

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults

GOOGLE_CIVIC_API_KEY = os.environ.get("GOOGLE_CIVIC_API_KEY")
CIVIC_API_URL = "https://www.googleapis.com/civicinfo/v2/voterinfo"


class CivicLookupInput(BaseModel):
    address: str = Field(
        ...,
        description="Full street address, including city, state, and zip.",
    )


@tool("civic_lookup", args_schema=CivicLookupInput)
def civic_lookup(address: str) -> dict:
    """
    Look up the contests (races) on a person's ballot for their address,
    using the Google Civic Information API. Returns each contest's office
    name, level (e.g. county, state, federal), district, and any candidates
    listed. Use this first for any question about someone's ballot,
    upcoming elections, or "what's on my ballot" — and to establish their
    jurisdiction (city/county/state) before calling office_responsibilities.
    """
    if not GOOGLE_CIVIC_API_KEY:
        return {"error": "GOOGLE_CIVIC_API_KEY is not set."}

    params = {"key": GOOGLE_CIVIC_API_KEY, "address": address}

    try:
        resp = requests.get(CIVIC_API_URL, params=params, timeout=10)
    except requests.RequestException as exc:
        return {"error": f"Request to Civic API failed: {exc}"}

    if resp.status_code != 200:
        return {
            "error": f"Civic API request failed with status {resp.status_code}",
            "detail": resp.text[:500],
        }

    data = resp.json()
    election = data.get("election", {})

    contests = []
    for contest in data.get("contests", []):
        contests.append(
            {
                "office": contest.get("office"),
                "level": contest.get("level"),
                "district": (contest.get("district") or {}).get("name"),
                "candidates": [c.get("name") for c in contest.get("candidates", [])],
            }
        )

    return {
        "election_name": election.get("name"),
        "election_day": election.get("electionDay"),
        "normalized_address": data.get("normalizedInput"),
        "contests": contests,
    }


class OfficeResponsibilitiesInput(BaseModel):
    office_name: str = Field(
        ...,
        description=(
            "The exact office/title, e.g. 'Board of Commissioners', "
            "'Sheriff', 'School Board Member'."
        ),
    )
    jurisdiction: str = Field(
        ...,
        description=(
            "The FULL jurisdiction chain for this office, as specific as "
            "possible — e.g. 'Douglas County, Georgia' or 'Douglasville, "
            "Douglas County, Georgia'. Never pass just a city or county name "
            "alone if you have more of the chain available; office names "
            "like 'Board of Commissioners' exist in thousands of places."
        ),
    )


@tool("office_responsibilities", args_schema=OfficeResponsibilitiesInput)
def office_responsibilities(office_name: str, jurisdiction: str) -> dict:
    """
    Search the web for what a specific office is responsible for, scoped
    tightly to its jurisdiction so it isn't confused with a same-named
    office elsewhere. Tries official .gov sources first, then falls back
    to a general search if that turns up nothing. Returns raw results
    (titles, URLs, snippets) for you to read and summarize yourself — do
    not treat an empty result as license to guess from the office's name
    alone; say plainly that you couldn't verify it instead.
    """
    search = DuckDuckGoSearchResults(output_format="list", num_results=6)

    query_gov = f"{office_name} {jurisdiction} responsibilities site:.gov"
    try:
        results = search.invoke(query_gov)
    except Exception as exc:
        return {"error": f"Search failed: {exc}"}

    used_query = query_gov
    if not results:
        query_general = f"{office_name} {jurisdiction} responsibilities duties"
        results = search.invoke(query_general)
        used_query = query_general

    if not results:
        return {
            "office_name": office_name,
            "jurisdiction": jurisdiction,
            "query_used": used_query,
            "results": [],
            "note": (
                "No search results found. Do not guess based on the "
                "office's name alone — tell the user you couldn't verify "
                "what this office controls."
            ),
        }

    return {
        "office_name": office_name,
        "jurisdiction": jurisdiction,
        "query_used": used_query,
        "results": results,
    }