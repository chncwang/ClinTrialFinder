import json
from urllib.parse import urlencode

import scrapy


class ClinicalTrialsSpider(scrapy.Spider):
    name = "clinical_trials"
    api_base_url = "https://clinicaltrials.gov/api/v2/studies"

    custom_settings = {
        "DOWNLOAD_DELAY": 1,
        "ROBOTSTXT_OBEY": False,
        "USER_AGENT": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "LOG_LEVEL": "INFO",
        "HTTPERROR_ALLOWED_CODES": [400, 401, 403, 404, 500],
    }

    def __init__(
        self,
        exclude_completed=False,
        condition=None,
        specific_trial=None,
        output_file=None,
        *args,
        **kwargs,
    ):
        super(ClinicalTrialsSpider, self).__init__(*args, **kwargs)
        self.exclude_completed = exclude_completed
        self.condition = condition
        self.specific_trial = specific_trial
        self.output_file = output_file
        self.logger.info(f"Exclude completed trials: {self.exclude_completed}")
        self.logger.info(f"Searching for condition: {self.condition}")
        if specific_trial:
            self.logger.info(f"Fetching specific trial: {specific_trial}")

    def get_base_params(self):
        """Get common request parameters"""
        params = {
            "format": "json",
            "fields": "NCTId,BriefTitle,OfficialTitle,Acronym,OrgStudyId,OverallStatus,StartDate,CompletionDate,PrimaryCompletionDate,BriefSummary,DetailedDescription,Condition,Keyword,StudyType,Phase,EnrollmentCount,MinimumAge,MaximumAge,Sex,HealthyVolunteers,EligibilityCriteria,LeadSponsorName,CollaboratorName,LocationFacility,LocationCity,LocationState,LocationCountry,LocationStatus,ArmName,ArmType,ArmDescription,InterventionName",
            "markupFormat": "markdown",
        }

        if not self.specific_trial:
            params.update(
                {
                    "query.cond": self.condition,
                    "pageSize": 100,
                    "countTotal": "true",
                }
            )

            if self.exclude_completed:
                params["filter.overallStatus"] = "NOT_YET_RECRUITING|RECRUITING"

        return params

    def create_request(self, url, callback):
        """Create a standardized request object"""
        return scrapy.Request(
            url=url,
            callback=callback,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            errback=self.handle_error,
            dont_filter=True,
        )

    def start_requests(self):
        params = self.get_base_params()

        if self.specific_trial:
            url = f"{self.api_base_url}/{self.specific_trial}?{urlencode(params)}"
            yield self.create_request(url, self.parse_single_trial)
        else:
            url = f"{self.api_base_url}?{urlencode(params)}"
            yield self.create_request(url, self.parse)

    def safe_get(self, d, *keys, default=None):
        """Safely get nested dictionary values"""
        for key in keys:
            if not isinstance(d, dict):
                return default
            d = d.get(key, default)
            if d is None:
                return default
        return d

    def parse(self, response):
        try:
            if response.status >= 400:
                self.logger.error(f"Error response {response.status}: {response.text}")
                return

            data = json.loads(response.text)
            studies = data.get("studies", [])

            self.logger.info(f"Found {len(studies)} studies on current page")
            if data.get("totalCount"):
                self.logger.info(f"Total studies available: {data.get('totalCount')}")

            for study in studies:
                # Debug logging
                self.logger.info(f"Study keys: {list(study.keys())}")
                self.logger.info(f"Study data: {study}")
                
                yield self.extract_trial_data(study)

            # Handle pagination
            next_page_token = data.get("nextPageToken")
            if next_page_token:
                params = self.get_base_params()
                params["pageToken"] = next_page_token
                next_url = f"{self.api_base_url}?{urlencode(params)}"
                yield self.create_request(next_url, self.parse)

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            self.logger.error(f"Response text: {response.text[:200]}...")
        except Exception as e:
            self.logger.error(f"Unexpected error while processing response: {e}")
            import traceback

            self.logger.error(f"Traceback: {traceback.format_exc()}")

    def handle_error(self, failure):
        self.logger.error(f"Request failed: {failure.value}")

    def closed(self, reason):
        self.logger.info(f"Spider closed: {reason}")

    def parse_single_trial(self, response):
        """Parse response for a single trial request."""
        try:
            if response.status >= 400:
                self.logger.error(f"Error response {response.status}: {response.text}")
                return

            data = json.loads(response.text)

            if data:
                trial_data = self.extract_trial_data(data)
                if self.output_file:
                    with open(self.output_file, "w") as f:
                        json.dump([trial_data], f)
                return trial_data
            else:
                self.logger.error(
                    f"Unexpected data structure. Available keys: {list(data.keys())}"
                )
                self.logger.error(f"Full response: {response.text}")

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            self.logger.error(f"Response text: {response.text[:200]}...")
        except Exception as e:
            self.logger.error(f"Unexpected error while processing response: {e}")
            import traceback

            self.logger.error(f"Traceback: {traceback.format_exc()}")

    def extract_trial_data(self, study_data, arms_interventions=None):
        """Extract trial data from flattened study data."""
        if arms_interventions is None:
            arms_interventions = {}
            
        # Extract arms information from the flattened structure
        arms = []
        arm_names = study_data.get("armName", [])
        arm_types = study_data.get("armType", [])
        arm_descriptions = study_data.get("armDescription", [])
        intervention_names = study_data.get("interventionName", [])
        
        # Group arms by their index
        max_arms = max(len(arm_names), len(arm_types), len(arm_descriptions))
        for i in range(max_arms):
            arm = {
                "name": arm_names[i] if i < len(arm_names) else None,
                "type": arm_types[i] if i < len(arm_types) else None,
                "description": arm_descriptions[i] if i < len(arm_descriptions) else None,
                "interventions": [intervention_names[i]] if i < len(intervention_names) and intervention_names[i] else [],
            }
            arms.append(arm)
            
        return {
            "identification": {
                "nct_id": study_data.get("nctId"),
                "url": (
                    f"https://clinicaltrials.gov/study/{study_data.get('nctId')}"
                    if study_data.get("nctId")
                    else None
                ),
                "brief_title": study_data.get("briefTitle"),
                "official_title": study_data.get("officialTitle"),
                "acronym": study_data.get("acronym"),
                "org_study_id": study_data.get("orgStudyId"),
            },
            "status": {
                "overall_status": study_data.get("overallStatus"),
                "start_date": study_data.get("startDate"),
                "completion_date": study_data.get("completionDate"),
                "primary_completion_date": study_data.get("primaryCompletionDate"),
            },
            "description": {
                "brief_summary": study_data.get("briefSummary"),
                "detailed_description": study_data.get("detailedDescription"),
                "conditions": study_data.get("condition"),
                "keywords": study_data.get("keyword"),
            },
            "design": {
                "study_type": study_data.get("studyType"),
                "phases": study_data.get("phase", []),
                "enrollment": study_data.get("enrollmentCount"),
                "arms": arms,
            },
            "eligibility": {
                "criteria": study_data.get("eligibilityCriteria"),
                "gender": study_data.get("sex"),
                "minimum_age": study_data.get("minimumAge"),
                "maximum_age": study_data.get("maximumAge"),
                "healthy_volunteers": study_data.get("healthyVolunteers"),
            },
            "contacts_locations": {
                "locations": [
                    {
                        "facility": study_data.get("locationFacility"),
                        "city": study_data.get("locationCity"),
                        "state": study_data.get("locationState"),
                        "country": study_data.get("locationCountry"),
                        "status": study_data.get("locationStatus"),
                    }
                ],
            },
            "sponsor": {
                "lead_sponsor": study_data.get("leadSponsorName"),
                "collaborators": study_data.get("collaboratorName", []),
            },
        }
