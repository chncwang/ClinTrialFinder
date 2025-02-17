# ClinTrialFinder

ClinTrialFinder is a sophisticated tool for downloading, filtering, and analyzing clinical trials data from ClinicalTrials.gov. It combines web crawling capabilities with intelligent filtering using GPT-4-mini to help researchers and medical professionals find relevant clinical trials. The tool accepts natural language descriptions of conditions (e.g. "early stage breast cancer in women over 50") and uses GPT-4-mini to evaluate these conditions against both the trials' titles and inclusion criteria to find relevant matches.

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
scrapy crawl clinical_trials -a condition="breast cancer" -a exclude_completed=true -o trials.json
```

Options:

- `condition`: Specific condition to search for
- `exclude_completed`: Set to `true` to exclude completed trials
- `-o trials.json`: Output file for the crawled data

### Filtering Trials

To filter trials based on specific conditions:

```bash
python -m scripts.filter_trials trials.json "breast cancer with bone metastases" "HER2 positive" "ECOG score is 1" \
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

The filtering process can be pipelined by using the output file from one filtering operation as the input for the next. For example:

```bash
# First filter for breast cancer trials with bone metastases
python -m scripts.filter_trials trials.json "breast cancer with bone metastases" --output bone_metastases_breast_cancer_trials.json

# Then filter those results for HER2 positive
python -m scripts.filter_trials bone_metastases_breast_cancer_trials.json "HER2 positive" --output her2_positive_trials.json

# Finally filter those results for ECOG score is 1
python -m scripts.filter_trials her2_positive_trials.json "ECOG score is 1" --output ecog_1_trials.json
```

This approach allows for incremental refinement of the trial set and can help break down complex filtering requirements into simpler steps.

## Output File Formats

The filtering process generates two JSON files:

1. **Filtered Trials** (`filtered_trials.json`): Contains the complete trial records that passed all filtering criteria
2. **Excluded Trials** (`filtered_trials_excluded.json`): Contains information about trials that were excluded and why they failed the filtering criteria

### Filtered Trials Format

The filtered trials JSON file contains the complete trial records that passed all criteria checks. Each trial entry preserves all fields from the original ClinicalTrials.gov data structure.

### Excluded Trials Format

The excluded trials JSON file contains entries for trials that failed the filtering criteria. Each entry includes:

#### Common Fields

- `nct_id`: The ClinicalTrials.gov identifier
- `brief_title`: The trial's brief title
- `eligibility_criteria`: The complete eligibility criteria text
- `failure_type`: The type of failure ("title" or "inclusion_criterion")
- `failure_message`: A general message about why the trial was excluded

#### Title-Based Exclusion Example

When a trial is excluded based on title evaluation:

```json
{
    "nct_id": "NCT05020860",
    "brief_title": "Correlation of Clinical Response to Pathologic Response in Patients With Early Breast Cancer",
    "eligibility_criteria": "...",
    "failure_type": "title",
    "failure_message": "Title check failed: The trial title focuses on early breast cancer and the correlation of clinical response to pathologic response, which does not specifically address patients with breast cancer that has metastasized to the bone, HER2 positive status, or an ECOG score of 1. Therefore, it is not suitable for the specified patient conditions."
}
```

#### Inclusion Criteria-Based Exclusion Example

When a trial is excluded based on inclusion criteria evaluation, additional fields are included:

```json
{
    "nct_id": "NCT04561362",
    "brief_title": "Study BT8009-100 in Subjects With Nectin-4 Expressing Advanced Malignancies",
    "eligibility_criteria": "...",
    "failure_type": "inclusion_criterion",
    "failure_message": "Failed inclusion criterion evaluation",
    "failed_condition": "HER2 positive",
    "failed_criterion": "Patients with locally advanced (unresectable) or metastatic, histologically confirmed breast cancer, either TNBC or hormone receptor (HR) positive and HER-2 negative according to ASCO/CAP guidelines and up to 3 prior lines of therapy for advanced (unresectable) or metastatic disease.",
    "failure_details": "Failed all OR branches:\nBranch 1: The inclusion criterion specifies patients with histologically confirmed breast cancer, specifically mentioning triple-negative breast cancer (TNBC). HER2 positive breast cancer does not fall under the TNBC category, which makes this inclusion criterion incompatible with the patient's condition.\nBranch 2: The inclusion criterion specifies that patients must have breast cancer that is hormone receptor positive and HER-2 negative. Since the patient condition is HER2 positive, it does not meet the inclusion criterion."
}
```

For inclusion criteria failures, these additional fields provide detailed information:

- `failed_condition`: The specific condition that failed to meet the criteria
- `failed_criterion`: The exact criterion that caused the failure
- `failure_details`: Detailed explanation of why the condition failed to meet the criterion, including analysis of different branches for OR-type criteria

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

## Future Work

Future work will:

1. Add support for exclusion criteria evaluation and reporting, enabling comprehensive trial eligibility assessment
2. Rank the filtered trials by expected outcome by analyzing metrics like PFS (Progression-Free Survival) and ORR (Objective Response Rate) from previous phases of the same drug/treatment to help prioritize more promising trials
3. Implement title and criteria vectorization for fast semantic search, enabling rapid trial filtering without repeated GPT API calls
4. Leverage GPT to generate optimized search keywords from patient conditions, enabling more efficient and targeted trial discovery

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
2. The GPT-4-mini filtering system, while sophisticated, may occasionally:
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

This tool should not be used as a primary source for medical decision-making.

BY USING THIS SOFTWARE, YOU ACKNOWLEDGE AND AGREE TO THESE TERMS AND LIMITATIONS.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Uses the ClinicalTrials.gov API v2
- Powered by OpenAI's GPT-4-mini
- Built with Scrapy for web crawling
