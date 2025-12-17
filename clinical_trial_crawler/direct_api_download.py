"""
Direct API client for ClinicalTrials.gov API v2.
Replaces Scrapy spider to avoid 403 blocking issues.
"""
import json
import logging
import requests
import time
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


class ClinicalTrialsAPIClient:
    """Direct HTTP client for ClinicalTrials.gov API v2"""
    
    def __init__(self):
        self.api_base_url = "https://clinicaltrials.gov/api/v2/studies"
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        })
        
    def get_base_params(self, condition: Optional[str] = None, exclude_completed: bool = False) -> Dict[str, Any]:
        """Get common request parameters"""
        params: Dict[str, Any] = {
            "format": "json",
            "fields": "ProtocolSection",
            "markupFormat": "markdown",
        }
        
        if condition:
            params.update({
                "query.cond": condition,
                "pageSize": "100",
                "countTotal": "true",
            })
            
            if exclude_completed:
                params["filter.overallStatus"] = "NOT_YET_RECRUITING|RECRUITING"
        
        return params
    
    def safe_get(self, d: Any, *keys: str, default: Any = None) -> Any:
        """Safely get nested dictionary values"""
        for key in keys:
            if not isinstance(d, dict):
                return default
            d = d.get(key, default)
            if d is None:
                return default
        return d
    
    def extract_trial_data(self, protocol: Dict[str, Any]) -> Dict[str, Any]:
        """Extract trial data from protocol section."""
        return {
            "identification": {
                "nct_id": self.safe_get(protocol, "identificationModule", "nctId"),
                "url": (
                    f"https://clinicaltrials.gov/study/{self.safe_get(protocol, 'identificationModule', 'nctId')}"
                    if self.safe_get(protocol, "identificationModule", "nctId")
                    else None
                ),
                "brief_title": self.safe_get(
                    protocol, "identificationModule", "briefTitle"
                ),
                "official_title": self.safe_get(
                    protocol, "identificationModule", "officialTitle"
                ),
                "acronym": self.safe_get(protocol, "identificationModule", "acronym"),
                "org_study_id": self.safe_get(
                    protocol, "identificationModule", "orgStudyIdInfo", "id"
                ),
            },
            "status": {
                "overall_status": self.safe_get(
                    protocol, "statusModule", "overallStatus"
                ),
                "start_date": self.safe_get(
                    protocol, "statusModule", "startDateStruct", "date"
                ),
                "completion_date": self.safe_get(
                    protocol, "statusModule", "completionDateStruct", "date"
                ),
                "primary_completion_date": self.safe_get(
                    protocol, "statusModule", "primaryCompletionDateStruct", "date"
                ),
            },
            "description": {
                "brief_summary": self.safe_get(
                    protocol, "descriptionModule", "briefSummary"
                ),
                "detailed_description": self.safe_get(
                    protocol, "descriptionModule", "detailedDescription"
                ),
                "conditions": self.safe_get(
                    protocol, "descriptionModule", "conditions"
                ),
                "keywords": self.safe_get(protocol, "descriptionModule", "keywords"),
            },
            "design": {
                "study_type": self.safe_get(protocol, "designModule", "studyType"),
                "phases": self.safe_get(protocol, "designModule", "phases", default=[]),
                "enrollment": self.safe_get(
                    protocol, "designModule", "enrollmentInfo", "count"
                ),
                "arms": [
                    {
                        "name": self.safe_get(arm, "label"),
                        "type": self.safe_get(arm, "type"),
                        "description": self.safe_get(arm, "description"),
                        "interventions": self.safe_get(
                            arm, "interventionNames", default=[]
                        ),
                    }
                    for arm in self.safe_get(
                        protocol, "armsInterventionsModule", "armGroups", default=[]
                    )
                ],
            },
            "eligibility": {
                "criteria": self.safe_get(
                    protocol, "eligibilityModule", "eligibilityCriteria"
                ),
                "gender": self.safe_get(protocol, "eligibilityModule", "sex"),
                "minimum_age": self.safe_get(
                    protocol, "eligibilityModule", "minimumAge"
                ),
                "maximum_age": self.safe_get(
                    protocol, "eligibilityModule", "maximumAge"
                ),
                "healthy_volunteers": self.safe_get(
                    protocol, "eligibilityModule", "healthyVolunteers"
                ),
            },
            "contacts_locations": {
                "locations": [
                    {
                        "facility": self.safe_get(loc, "facility", "name") or self.safe_get(loc, "name"),
                        "city": self.safe_get(loc, "facility", "city") or self.safe_get(loc, "city"),
                        "state": self.safe_get(loc, "facility", "state") or self.safe_get(loc, "state"),
                        "country": self.safe_get(loc, "facility", "country") or self.safe_get(loc, "country"),
                        "status": self.safe_get(loc, "status"),
                    }
                    for loc in self.safe_get(
                        protocol, "contactsLocationsModule", "locations", default=[]
                    )
                ],
            },
            "sponsor": {
                "lead_sponsor": self.safe_get(
                    protocol, "sponsorCollaboratorsModule", "leadSponsor", "name"
                ),
                "collaborators": [
                    self.safe_get(collab, "name", default="")
                    for collab in self.safe_get(
                        protocol,
                        "sponsorCollaboratorsModule",
                        "collaborators",
                        default=[],
                    )
                ],
            },
        }
    
    def download_trials(
        self,
        condition: Optional[str] = None,
        exclude_completed: bool = False,
        specific_trial: Optional[str] = None,
        output_file: Optional[str] = None
    ) -> bool:
        """Download trials and save to file"""
        
        all_trials: List[Dict[str, Any]] = []
        
        try:
            if specific_trial:
                # Download single trial
                params = self.get_base_params()
                url = f"{self.api_base_url}/{specific_trial}?{urlencode(params)}"
                
                logger.info(f"Downloading specific trial: {specific_trial}")
                response = self.session.get(url, timeout=30)
                
                if response.status_code != 200:
                    logger.error(f"Error {response.status_code}: {response.text[:200]}")
                    return False
                
                data = response.json()
                if "protocolSection" in data:
                    trial_data = self.extract_trial_data(data["protocolSection"])
                    all_trials.append(trial_data)
                    
            else:
                # Download trials by condition
                params = self.get_base_params(condition, exclude_completed)
                next_page_token = None
                page_count = 0
                
                while True:
                    page_count += 1
                    if next_page_token:
                        params["pageToken"] = next_page_token
                    
                    url = f"{self.api_base_url}?{urlencode(params)}"
                    logger.info(f"Downloading page {page_count}...")
                    
                    response = self.session.get(url, timeout=30)
                    
                    if response.status_code != 200:
                        logger.error(f"Error {response.status_code}: {response.text[:200]}")
                        break
                    
                    data = response.json()
                    studies = data.get("studies", [])
                    
                    logger.info(f"Found {len(studies)} studies on page {page_count}")
                    if data.get("totalCount"):
                        logger.info(f"Total studies available: {data.get('totalCount')}")
                    
                    for study in studies:
                        if "protocolSection" in study:
                            trial_data = self.extract_trial_data(study["protocolSection"])
                            all_trials.append(trial_data)
                    
                    # Check for next page
                    next_page_token = data.get("nextPageToken")
                    if not next_page_token:
                        break
                    
                    # Rate limiting - 1 second delay between requests
                    time.sleep(1)
            
            # Save to file
            if output_file:
                with open(output_file, "w") as f:
                    json.dump(all_trials, f, indent=2)
                logger.info(f"Saved {len(all_trials)} trials to {output_file}")
            
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    # Test the client
    logging.basicConfig(level=logging.INFO)
    client = ClinicalTrialsAPIClient()
    client.download_trials(
        condition="nasopharyngeal carcinoma",
        exclude_completed=True,
        output_file="test_output.json"
    )
