import scrapy


class ClinicalTrialsSpider(scrapy.Spider):
    name = "clinical_trials"
    start_urls = ["https://clinicaltrials.gov/search?cond=Nasopharyngeal%20Carcinoma"]

    custom_settings = {
        "DOWNLOAD_DELAY": 2,  # Add 2 second delay between requests
        "ROBOTSTXT_OBEY": False,  # Temporarily disable for testing
        "USER_AGENT": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "LOG_LEVEL": "DEBUG",  # Add this line to enable debug logging
    }

    def parse(self, response):
        self.logger.debug(f"Response status: {response.status}")
        self.logger.debug(f"Response headers: {response.headers}")
        self.logger.debug(f"Response body preview: {response.body}")
        title_links = response.css("a.hit-card-title")

        for link in title_links:
            title_parts = link.css("::text, mark::text").getall()
            full_title = "".join(title_parts).strip()
            self.logger.info(f"Found study title: {full_title}")
