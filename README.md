# ClinTrialFinder

ClinTrialFinder is a sophisticated tool for downloading, filtering, and analyzing clinical trials data from ClinicalTrials.gov. It combines web crawling capabilities with intelligent filtering using GPT-4-mini to help researchers and medical professionals find relevant clinical trials based on specific conditions and criteria.

## Features

- **Automated Data Collection**: Crawls ClinicalTrials.gov using their API v2 to fetch trial data based on disease name
- **Smart Filtering**: Uses GPT-4-mini to evaluate trial eligibility based on:
  - Trial titles
  - Inclusion criteria
- **Flexible Search Options**: Filter trials by:
  - Recruitment status
  - Trial phase
  - Study type
  - Custom conditions and criteria
- **Structured Data Model**: Comprehensive data classes for representing clinical trial information
- **Caching System**: Implements response caching to optimize API usage and reduce costs

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/ClinTrialFinder.git
   cd ClinTrialFinder
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:

   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

## Usage

### Crawling Clinical Trials

To download clinical trials data:

```bash
cd clinical_trial_crawler
scrapy crawl clinical_trials -a condition="cancer" -a exclude_completed=true -o trials.json
```

Options:

- `condition`: Specific condition to search for
- `exclude_completed`: Set to `true` to exclude completed trials
- `-o trials.json`: Output file for the crawled data

### Filtering Trials

To filter trials based on specific conditions:

```bash
python scripts/filter_trials.py trials.json "stage IV breast cancer" "HER2 positive" \
    --recruiting \
    --phase 2 \
    --exclude-study-type Observational \
    --output filtered_trials.json
```

Options:

- `--recruiting`: Filter for only recruiting trials
- `--phase`: Filter by trial phase (1-4)
- `--exclude-study-type`: Exclude specific study types
- `--output`: Specify output file path
- `--cache-size`: Set maximum number of cached responses (default: 10000)
- `--api-key`: Provide OpenAI API key (alternatively, use OPENAI_API_KEY environment variable)

## GPT-4-mini Integration

The system uses GPT-4-mini to:

1. Evaluate trial titles against conditions
2. Parse and split inclusion criteria
3. Evaluate individual criteria against conditions
4. Handle complex OR/AND logic in criteria
Responses are cached to optimize API usage and reduce costs.

## Logging

When executing `scripts/filter_trials.py`, the system maintains detailed logs including:

- Trial processing progress
- GPT-4 API costs
- Eligibility decisions and reasons
- Error messages and debugging information
Log files are created with timestamps in the format: `filter_trials_YYYYMMDD_HHMMSS.log`

## Contributing

### Submitting Changes

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Reporting Issues

Found a bug or have a suggestion? Open a new issue with a clear title and description.

## Disclaimer

**IMPORTANT: Please read this disclaimer carefully before using ClinTrialFinder**

This software is provided for research and informational purposes only. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition or clinical trial participation.

### Important Notes

1. The information retrieved by this tool from ClinicalTrials.gov may be incomplete, outdated, or inaccurate.
2. The GPT-4 filtering system, while sophisticated, may occasionally:
   - Miss relevant trials
   - Include irrelevant trials
   - Misinterpret eligibility criteria
3. Clinical trial eligibility can only be definitively determined by the trial's medical team.
4. This tool does not provide medical advice or recommendations.
5. Users should independently verify all information obtained through this tool.
6. The developers are not responsible for any decisions made based on the output of this tool.

### Data Privacy Notice

- User queries and filtered results may be processed through third-party APIs (OpenAI).
- Users should not input personally identifiable health information.
- Review OpenAI's privacy policy regarding data handling.

### Regulatory Compliance

This tool is not FDA-approved and should not be used as a primary source for medical decision-making.

BY USING THIS SOFTWARE, YOU ACKNOWLEDGE AND AGREE TO THESE TERMS AND LIMITATIONS.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Uses the ClinicalTrials.gov API v2
- Powered by OpenAI's GPT-4-mini
- Built with Scrapy for web crawling
