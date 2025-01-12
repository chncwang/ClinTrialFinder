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
    }

    def start_requests(self):
        # API query parameters
        params = {
            "query.cond": "Nasopharyngeal Carcinoma",
            "format": "json",
            "pageSize": 100,  # Number of results per page
            "page": 1,
        }

        url = f"{self.api_base_url}?{urlencode(params)}"
        yield scrapy.Request(
            url=url, callback=self.parse, headers={"Accept": "application/json"}
        )

    def parse(self, response):
        try:
            # Parse JSON response
            data = json.loads(response.body)
            studies = data.get("studies", [])

            # Extract and yield study information
            for study in studies:
                protocol = study.get("protocolSection", {})
                identification = protocol.get("identificationModule", {})
                status = protocol.get("statusModule", {})

                yield {
                    "nct_id": identification.get("nctId"),
                    "title": identification.get("officialTitle"),
                    "brief_title": identification.get("briefTitle"),
                    "status": status.get("overallStatus"),
                    "phase": protocol.get("designModule", {}).get("phases", []),
                    "start_date": status.get("startDateStruct", {}).get("date"),
                    "completion_date": status.get("completionDateStruct", {}).get(
                        "date"
                    ),
                }

            # Handle pagination
            total_count = data.get("totalCount", 0)
            current_page = data.get("page", 1)
            page_size = data.get("pageSize", 100)

            if (current_page * page_size) < total_count:
                next_page = current_page + 1
                params = {
                    "query.cond": "Nasopharyngeal Carcinoma",
                    "format": "json",
                    "pageSize": page_size,
                    "page": next_page,
                }
                next_url = f"{self.api_base_url}?{urlencode(params)}"

                yield scrapy.Request(
                    url=next_url,
                    callback=self.parse,
                    headers={"Accept": "application/json"},
                )

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error while processing response: {e}")

    def closed(self, reason):
        self.logger.info(f"Spider closed: {reason}")
