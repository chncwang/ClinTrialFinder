import json
from urllib.parse import urlencode

import scrapy
from scrapy.utils.project import get_project_settings


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

    def __init__(self, exclude_completed=False, *args, **kwargs):
        super(ClinicalTrialsSpider, self).__init__(*args, **kwargs)
        self.exclude_completed = exclude_completed
        self.logger.info(f"Exclude completed trials: {self.exclude_completed}")

    def start_requests(self):
        # Requesting comprehensive set of fields
        fields = [
            # Identification
            "IdentificationModule",
            # Study Overview
            "DescriptionModule",
            # Status and Dates
            "StatusModule",
            # Study Design
            "DesignModule",
            # Eligibility
            "EligibilityModule",
            # Contacts and Locations
            "ContactsLocationsModule",
            # Sponsor/Collaborators
            "SponsorCollaboratorsModule",
            # Oversight
            "OversightModule",
        ]

        params = {
            "query.cond": "Nasopharyngeal Carcinoma",
            "format": "json",
            "pageSize": 100,
            "countTotal": "true",
            "fields": ",".join(fields),
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

    def extract_module_data(self, module_dict, fields):
        """Helper function to safely extract nested fields"""
        result = {}
        for field in fields:
            value = module_dict.get(field)
            if isinstance(value, dict):
                # Handle date structures
                if "date" in value:
                    value = value.get("date")
            result[field] = value
        return result

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
                protocol = study.get("protocolSection", {})

                # Extract data from each module
                identification = protocol.get("identificationModule", {})
                description = protocol.get("descriptionModule", {})
                status = protocol.get("statusModule", {})
                design = protocol.get("designModule", {})
                eligibility = protocol.get("eligibilityModule", {})
                contacts_locations = protocol.get("contactsLocationsModule", {})
                sponsor = protocol.get("sponsorCollaboratorsModule", {})
                oversight = protocol.get("oversightModule", {})

                yield {
                    "identification": {
                        "nct_id": identification.get("nctId"),
                        "org_study_id": identification.get("orgStudyIdInfo", {}).get(
                            "id"
                        ),
                        "brief_title": identification.get("briefTitle"),
                        "official_title": identification.get("officialTitle"),
                        "acronym": identification.get("acronym"),
                    },
                    "status_info": {
                        "status": status.get("overallStatus"),
                        "start_date": status.get("startDateStruct", {}).get("date"),
                        "completion_date": status.get("completionDateStruct", {}).get(
                            "date"
                        ),
                        "primary_completion_date": status.get(
                            "primaryCompletionDateStruct", {}
                        ).get("date"),
                        "last_update_date": status.get(
                            "lastUpdatePostDateStruct", {}
                        ).get("date"),
                    },
                    "study_overview": {
                        "brief_summary": description.get("briefSummary"),
                        "detailed_description": description.get("detailedDescription"),
                        "conditions": description.get("conditions"),
                        "keywords": description.get("keywords"),
                    },
                    "design_info": {
                        "study_type": design.get("studyType"),
                        "phases": design.get("phases", []),
                        "design_info": design.get("designInfo"),
                        "enrollment_count": design.get("enrollmentInfo", {}).get(
                            "count"
                        ),
                        "arms": [
                            {
                                "name": arm.get("name"),
                                "description": arm.get("description"),
                                "type": arm.get("type"),
                                "intervention_names": arm.get("interventionNames", []),
                            }
                            for arm in design.get("arms", [])
                        ],
                    },
                    "eligibility_criteria": {
                        "criteria_text": eligibility.get("eligibilityCriteria"),
                        "gender": eligibility.get("sex"),
                        "minimum_age": eligibility.get("minimumAge"),
                        "maximum_age": eligibility.get("maximumAge"),
                        "healthy_volunteers": eligibility.get("healthyVolunteers"),
                    },
                    "locations": [
                        {
                            "facility": loc.get("facility", {}).get("name"),
                            "city": loc.get("facility", {}).get("city"),
                            "state": loc.get("facility", {}).get("state"),
                            "country": loc.get("facility", {}).get("country"),
                            "status": loc.get("status"),
                        }
                        for loc in contacts_locations.get("locations", [])
                    ],
                    "sponsor_info": {
                        "lead_sponsor": sponsor.get("leadSponsor", {}).get("name"),
                        "collaborators": [
                            collab.get("name")
                            for collab in sponsor.get("collaborators", [])
                        ],
                    },
                    "oversight_info": {
                        "has_dmc": oversight.get("hasDmc"),
                        "is_fda_regulated": oversight.get("isFdaRegulatedDrug")
                        or oversight.get("isFdaRegulatedDevice"),
                        "is_section_801": oversight.get("isUnapprovedDevice"),
                    },
                }

            # Handle pagination
            next_page_token = data.get("nextPageToken")
            if next_page_token:
                params = {
                    "query.cond": "Nasopharyngeal Carcinoma",
                    "format": "json",
                    "pageSize": 100,
                    "pageToken": next_page_token,
                    "fields": ",".join(fields),
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

    def handle_error(self, failure):
        self.logger.error(f"Request failed: {failure.value}")

    def closed(self, reason):
        self.logger.info(f"Spider closed: {reason}")
