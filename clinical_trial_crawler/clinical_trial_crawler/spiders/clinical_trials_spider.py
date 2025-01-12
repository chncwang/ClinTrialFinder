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
        "HTTPERROR_ALLOWED_CODES": [
            400,
            401,
            403,
            404,
            500,
        ],  # Allow error codes to be processed
    }

    def __init__(self, exclude_completed=False, *args, **kwargs):
        super(ClinicalTrialsSpider, self).__init__(*args, **kwargs)
        self.exclude_completed = exclude_completed
        self.logger.info(f"Exclude completed trials: {self.exclude_completed}")

    def start_requests(self):
        # API query parameters based on documentation
        params = {
            "query.cond": "Nasopharyngeal Carcinoma",
            "format": "json",
            "pageSize": 100,
            "countTotal": "true",  # Get total count in first response
            "fields": "NCTId,BriefTitle,OfficialTitle,OverallStatus,Phase,StartDate,CompletionDate",
            "markupFormat": "markdown",  # Specify markup format as per API docs
        }

        # Add status filter if exclude-completed flag is set
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
            dont_filter=True,  # Allow duplicate requests if needed
        )

    def parse(self, response):
        try:
            # Check if response status is error
            if response.status >= 400:
                self.logger.error(f"Error response {response.status}: {response.text}")
                return

            # Parse JSON response
            data = json.loads(response.text)
            studies = data.get("studies", [])

            # Log study count for debugging
            self.logger.info(f"Found {len(studies)} studies on current page")
            if data.get("totalCount"):
                self.logger.info(f"Total studies available: {data.get('totalCount')}")

            # Extract and yield study information
            for study in studies:
                # Extract data using proper field paths based on API structure
                yield {
                    "nct_id": study.get("protocolSection", {})
                    .get("identificationModule", {})
                    .get("nctId"),
                    "brief_title": study.get("protocolSection", {})
                    .get("identificationModule", {})
                    .get("briefTitle"),
                    "official_title": study.get("protocolSection", {})
                    .get("identificationModule", {})
                    .get("officialTitle"),
                    "status": study.get("protocolSection", {})
                    .get("statusModule", {})
                    .get("overallStatus"),
                    "phase": study.get("protocolSection", {})
                    .get("designModule", {})
                    .get("phases", []),
                    "start_date": study.get("protocolSection", {})
                    .get("statusModule", {})
                    .get("startDateStruct", {})
                    .get("date"),
                    "completion_date": study.get("protocolSection", {})
                    .get("statusModule", {})
                    .get("completionDateStruct", {})
                    .get("date"),
                }

            # Handle pagination using nextPageToken
            next_page_token = data.get("nextPageToken")
            if next_page_token:
                params = {
                    "query.cond": "Nasopharyngeal Carcinoma",
                    "format": "json",
                    "pageSize": 100,
                    "pageToken": next_page_token,
                    "fields": "NCTId,BriefTitle,OfficialTitle,OverallStatus,Phase,StartDate,CompletionDate",
                    "markupFormat": "markdown",
                }

                # Add status filter if exclude-completed flag is set
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
            self.logger.error(
                f"Response text: {response.text[:200]}..."
            )  # Log first 200 chars of response
        except Exception as e:
            self.logger.error(f"Unexpected error while processing response: {e}")

    def handle_error(self, failure):
        """Handle request errors"""
        self.logger.error(f"Request failed: {failure.value}")

    def closed(self, reason):
        self.logger.info(f"Spider closed: {reason}")
