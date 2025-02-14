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
        *args,
        **kwargs,
    ):
        super(ClinicalTrialsSpider, self).__init__(*args, **kwargs)
        self.exclude_completed = exclude_completed
        self.condition = condition
        self.logger.info(f"Exclude completed trials: {self.exclude_completed}")
        self.logger.info(f"Searching for condition: {self.condition}")

    def start_requests(self):
        params = {
            "query.cond": self.condition,
            "format": "json",
            "pageSize": 100,
            "countTotal": "true",
            "fields": "ProtocolSection",  # Request entire ProtocolSection
            "markupFormat": "markdown",
        }

        if self.exclude_completed:
            params["filter.overallStatus"] = (
                "ACTIVE_NOT_RECRUITING|ENROLLING_BY_INVITATION|"
                "NOT_YET_RECRUITING|RECRUITING|SUSPENDED"
            )

        url = f"{self.api_base_url}?{urlencode(params)}"
        yield scrapy.Request(
            url=url,
            callback=self.parse,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            errback=self.handle_error,
            dont_filter=True,
        )

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
                protocol = self.safe_get(study, "protocolSection", default={})
                nct_id = self.safe_get(protocol, "identificationModule", "nctId")

                yield {
                    "identification": {
                        "nct_id": nct_id,
                        "url": (
                            f"https://clinicaltrials.gov/study/{nct_id}"
                            if nct_id
                            else None
                        ),
                        "brief_title": self.safe_get(
                            protocol, "identificationModule", "briefTitle"
                        ),
                        "official_title": self.safe_get(
                            protocol, "identificationModule", "officialTitle"
                        ),
                        "acronym": self.safe_get(
                            protocol, "identificationModule", "acronym"
                        ),
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
                            protocol,
                            "statusModule",
                            "primaryCompletionDateStruct",
                            "date",
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
                        "keywords": self.safe_get(
                            protocol, "descriptionModule", "keywords"
                        ),
                    },
                    "design": {
                        "study_type": self.safe_get(
                            protocol, "designModule", "studyType"
                        ),
                        "phases": self.safe_get(
                            protocol, "designModule", "phases", default=[]
                        ),
                        "enrollment": self.safe_get(
                            protocol, "designModule", "enrollmentInfo", "count"
                        ),
                        "arms": [
                            {
                                "name": self.safe_get(arm, "name"),
                                "type": self.safe_get(arm, "type"),
                                "description": self.safe_get(arm, "description"),
                                "interventions": self.safe_get(
                                    arm, "interventionNames", default=[]
                                ),
                            }
                            for arm in self.safe_get(
                                protocol, "designModule", "arms", default=[]
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
                                "facility": self.safe_get(loc, "facility", "name"),
                                "city": self.safe_get(loc, "facility", "city"),
                                "state": self.safe_get(loc, "facility", "state"),
                                "country": self.safe_get(loc, "facility", "country"),
                                "status": self.safe_get(loc, "status"),
                            }
                            for loc in self.safe_get(
                                protocol,
                                "contactsLocationsModule",
                                "locations",
                                default=[],
                            )
                        ],
                    },
                    "sponsor": {
                        "lead_sponsor": self.safe_get(
                            protocol,
                            "sponsorCollaboratorsModule",
                            "leadSponsor",
                            "name",
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

            # Handle pagination
            next_page_token = data.get("nextPageToken")
            if next_page_token:
                params = {
                    "query.cond": self.condition,
                    "format": "json",
                    "pageSize": 100,
                    "pageToken": next_page_token,
                    "fields": "ProtocolSection",
                    "markupFormat": "markdown",
                }

                if self.exclude_completed:
                    params["filter.overallStatus"] = (
                        "ACTIVE_NOT_RECRUITING|ENROLLING_BY_INVITATION|"
                        "NOT_YET_RECRUITING|RECRUITING|SUSPENDED"
                    )

                next_url = f"{self.api_base_url}?{urlencode(params)}"
                yield scrapy.Request(
                    url=next_url,
                    callback=self.parse,
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                    },
                    errback=self.handle_error,
                    dont_filter=True,
                )

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
