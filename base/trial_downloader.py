"""
Module for downloading clinical trials from ClinicalTrials.gov API v2.

Replaces the Scrapy-based crawler with direct requests-based API calls.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

API_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


def safe_get(d: Any, *keys: str, default: Any = None) -> Any:
    """Safely get nested dictionary values."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, default)
        if d is None:
            return default
    return d


def extract_trial_data(protocol: Dict[str, Any]) -> Dict[str, Any]:
    """Extract trial data from a ClinicalTrials.gov protocol section.

    Args:
        protocol: The protocolSection dict from the API response.

    Returns:
        Structured trial data dictionary.
    """
    return {
        "identification": {
            "nct_id": safe_get(protocol, "identificationModule", "nctId"),
            "url": (
                f"https://clinicaltrials.gov/study/{safe_get(protocol, 'identificationModule', 'nctId')}"
                if safe_get(protocol, "identificationModule", "nctId")
                else None
            ),
            "brief_title": safe_get(
                protocol, "identificationModule", "briefTitle"
            ),
            "official_title": safe_get(
                protocol, "identificationModule", "officialTitle"
            ),
            "acronym": safe_get(protocol, "identificationModule", "acronym"),
            "org_study_id": safe_get(
                protocol, "identificationModule", "orgStudyIdInfo", "id"
            ),
        },
        "status": {
            "overall_status": safe_get(
                protocol, "statusModule", "overallStatus"
            ),
            "start_date": safe_get(
                protocol, "statusModule", "startDateStruct", "date"
            ),
            "completion_date": safe_get(
                protocol, "statusModule", "completionDateStruct", "date"
            ),
            "primary_completion_date": safe_get(
                protocol, "statusModule", "primaryCompletionDateStruct", "date"
            ),
        },
        "description": {
            "brief_summary": safe_get(
                protocol, "descriptionModule", "briefSummary"
            ),
            "detailed_description": safe_get(
                protocol, "descriptionModule", "detailedDescription"
            ),
            "conditions": safe_get(
                protocol, "descriptionModule", "conditions"
            ),
            "keywords": safe_get(protocol, "descriptionModule", "keywords"),
        },
        "design": {
            "study_type": safe_get(protocol, "designModule", "studyType"),
            "phases": safe_get(protocol, "designModule", "phases", default=[]),
            "enrollment": safe_get(
                protocol, "designModule", "enrollmentInfo", "count"
            ),
            "arms": [
                {
                    "name": safe_get(arm, "label"),
                    "type": safe_get(arm, "type"),
                    "description": safe_get(arm, "description"),
                    "interventions": safe_get(
                        arm, "interventionNames", default=[]
                    ),
                }
                for arm in safe_get(
                    protocol, "armsInterventionsModule", "armGroups", default=[]
                )
            ],
        },
        "eligibility": {
            "criteria": safe_get(
                protocol, "eligibilityModule", "eligibilityCriteria"
            ),
            "gender": safe_get(protocol, "eligibilityModule", "sex"),
            "minimum_age": safe_get(
                protocol, "eligibilityModule", "minimumAge"
            ),
            "maximum_age": safe_get(
                protocol, "eligibilityModule", "maximumAge"
            ),
            "healthy_volunteers": safe_get(
                protocol, "eligibilityModule", "healthyVolunteers"
            ),
        },
        "contacts_locations": {
            "locations": [
                {
                    "facility": safe_get(loc, "facility", "name") or safe_get(loc, "name"),
                    "city": safe_get(loc, "facility", "city") or safe_get(loc, "city"),
                    "state": safe_get(loc, "facility", "state") or safe_get(loc, "state"),
                    "country": safe_get(loc, "facility", "country") or safe_get(loc, "country"),
                    "status": safe_get(loc, "status"),
                }
                for loc in safe_get(
                    protocol, "contactsLocationsModule", "locations", default=[]
                )
            ],
        },
        "sponsor": {
            "lead_sponsor": safe_get(
                protocol, "sponsorCollaboratorsModule", "leadSponsor", "name"
            ),
            "collaborators": [
                safe_get(collab, "name", default="")
                for collab in safe_get(
                    protocol,
                    "sponsorCollaboratorsModule",
                    "collaborators",
                    default=[],
                )
            ],
        },
    }


class TrialDownloader:
    """Downloads clinical trials from ClinicalTrials.gov API v2 using requests."""

    def __init__(self, delay: float = 1.0, timeout: int = 30) -> None:
        self.delay = delay
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        })

    def _get_base_params(
        self,
        condition: Optional[str] = None,
        exclude_completed: bool = False,
        page_size: int = 100,
    ) -> Dict[str, Union[str, int]]:
        """Build common API request parameters."""
        params: Dict[str, Union[str, int]] = {
            "format": "json",
            "fields": "ProtocolSection",
            "markupFormat": "markdown",
        }

        if condition:
            params["query.cond"] = condition
            params["pageSize"] = str(page_size)
            params["countTotal"] = "true"

            if exclude_completed:
                params["filter.overallStatus"] = "NOT_YET_RECRUITING|RECRUITING"

        return params

    def fetch_by_condition(
        self,
        condition: str,
        exclude_completed: bool = False,
        page_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """Fetch all trials matching a condition, handling pagination.

        Args:
            condition: Disease or condition to search for.
            exclude_completed: If True, only return recruiting trials.
            page_size: Number of results per API page.

        Returns:
            List of trial data dictionaries.
        """
        params = self._get_base_params(condition, exclude_completed, page_size)
        all_trials: List[Dict[str, Any]] = []
        page = 1

        while True:
            url = f"{API_BASE_URL}?{urlencode(params)}"
            logger.info(f"Fetching page {page} for condition: {condition}")

            try:
                response = self.session.get(url, timeout=self.timeout)

                if response.status_code >= 400:
                    logger.error(f"Error response {response.status_code}: {response.text[:200]}")
                    break

                data = response.json()
                studies = data.get("studies", [])

                logger.info(f"Found {len(studies)} studies on page {page}")
                if data.get("totalCount"):
                    logger.info(f"Total studies available: {data.get('totalCount')}")

                for study in studies:
                    if "protocolSection" in study:
                        all_trials.append(extract_trial_data(study["protocolSection"]))

                # Handle pagination
                next_page_token = data.get("nextPageToken")
                if next_page_token:
                    params["pageToken"] = next_page_token
                    page += 1
                    time.sleep(self.delay)
                else:
                    break

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {e}")
                break
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                break

        logger.info(f"Downloaded {len(all_trials)} total trials for condition: {condition}")
        return all_trials

    def fetch_by_nct_id(self, nct_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single trial by NCT ID.

        Args:
            nct_id: The NCT identifier (e.g., "NCT04513678").

        Returns:
            Trial data dictionary, or None if not found.
        """
        params: Dict[str, Union[str, int]] = {
            "format": "json",
            "fields": "ProtocolSection",
            "markupFormat": "markdown",
        }

        url = f"{API_BASE_URL}/{nct_id}?{urlencode(params)}"
        logger.info(f"Fetching trial: {nct_id}")

        try:
            response = self.session.get(url, timeout=self.timeout)

            if response.status_code >= 400:
                logger.error(f"Error response {response.status_code} for {nct_id}: {response.text[:200]}")
                return None

            data = response.json()

            if "protocolSection" in data:
                return extract_trial_data(data["protocolSection"])
            else:
                logger.error(f"Unexpected data structure for {nct_id}. Keys: {list(data.keys())}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {nct_id}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response for {nct_id}: {e}")
            return None
