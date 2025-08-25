# GPT Categorization of False Positive Error Cases

This document describes the enhanced error case analysis functionality that uses GPT-5 (or GPT-4o as fallback) to automatically categorize false positive errors in clinical trial filtering systems.

## Overview

The enhanced error case analyzer can now automatically categorize false positive errors into three specific categories:

1. **Exclusion Criteria Violation**: The patient meets inclusion criteria but violates exclusion criteria
2. **Inclusion Criteria Violation**: The patient does not meet one or more inclusion criteria
3. **Data Label Error**: This is actually a true positive case with incorrect ground truth labeling

## Features

- **GPT-5 Integration**: Primary model for categorization with automatic fallback to GPT-4o
- **Automatic Categorization**: Processes all false positive cases in the dataset
- **CSV Export**: Exports categorized results with detailed case information
- **Comprehensive Logging**: Detailed progress tracking and error reporting
- **Rate Limiting Protection**: Built-in delays to avoid API rate limits

## Prerequisites

1. **OpenAI API Key**: Set the `OPENAI_API_KEY` environment variable
2. **Python Dependencies**: Ensure all required packages are installed
3. **Error Case JSON File**: Valid error case collection file to analyze

## Usage

### Basic Analysis (No GPT Categorization)

```bash
python -m scripts.analyze_error_cases results/error_cases_20250825_063056.json
```

### With GPT Categorization

```bash
python -m scripts.analyze_error_cases results/error_cases_20250825_063056.json --categorize-gpt
```

### Export Results to CSV

```bash
python -m scripts.analyze_error_cases results/error_cases_20250825_063056.json --categorize-gpt --export-csv
```

### Custom Output File

```bash
python -m scripts.analyze_error_cases results/error_cases_20250825_063056.json --categorize-gpt --output my_categorized_results.csv
```

## Test Script

A dedicated test script is available to verify the GPT categorization functionality:

```bash
python -m scripts.test_gpt_categorization results/error_cases_20250825_063056.json
```

## Output Format

The categorized results CSV contains the following columns:

- `patient_id`: Patient identifier
- `disease_name`: Disease/condition name
- `trial_id`: Clinical trial identifier
- `trial_title`: Trial title
- `trial_criteria`: Trial inclusion/exclusion criteria
- `suitability_probability`: Model's suitability score
- `reason`: Original error reason
- `text_summary`: Case summary text
- `gpt_categorization`: GPT-assigned error category
- `model_used`: GPT model used for categorization

## Error Categories

### 1. Exclusion Criteria Violation
- **Definition**: Patient meets inclusion criteria but violates exclusion criteria
- **Example**: Patient has the right cancer type and stage but is too old for the trial
- **Impact**: Trial filtering system correctly identified eligibility, but exclusion criteria were missed

### 2. Inclusion Criteria Violation
- **Definition**: Patient does not meet one or more inclusion criteria
- **Example**: Patient has wrong cancer type or disease stage
- **Impact**: Trial filtering system incorrectly identified eligibility when patient should be excluded

### 3. Data Label Error
- **Definition**: This is actually a true positive case with incorrect ground truth
- **Example**: Patient should be eligible but was labeled as ineligible in the dataset
- **Impact**: Trial filtering system is working correctly; the ground truth label is wrong

## Model Fallback Strategy

The system automatically tries models in this order:

1. **GPT-5**: Primary model (highest accuracy)
2. **GPT-4o**: Fallback model (high accuracy, more widely available)

If GPT-5 is not available or fails, the system automatically falls back to GPT-4o.

## Cost Considerations

Estimated costs per 1K tokens:
- **GPT-5**: $5.00 input, $15.00 output
- **GPT-4o**: $2.50 input, $10.00 output

The system uses minimal tokens by:
- Concise system prompts
- Structured user prompts
- Single-word responses (category names only)

## Error Handling

The system handles various error scenarios:

- **API Failures**: Automatic retry with fallback models
- **Rate Limiting**: Built-in delays between requests
- **Invalid Responses**: Logging and graceful degradation
- **Network Issues**: Comprehensive error logging

## Performance

- **Processing Speed**: ~0.1 second delay between cases to avoid rate limits
- **Batch Processing**: Processes all false positive cases sequentially
- **Memory Usage**: Minimal memory footprint for large datasets
- **API Efficiency**: Optimized prompts for minimal token usage

## Troubleshooting

### Common Issues

1. **API Key Not Set**
   ```
   Error: OPENAI_API_KEY environment variable not set
   Solution: Set your OpenAI API key: export OPENAI_API_KEY="your-key-here"
   ```

2. **Model Not Available**
   ```
   Warning: Failed to categorize with gpt-5: Model not found
   Solution: System automatically falls back to GPT-4o
   ```

3. **Rate Limiting**
   ```
   Warning: Rate limit exceeded
   Solution: System automatically adds delays between requests
   ```

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
python -m scripts.analyze_error_cases results/error_cases.json --categorize-gpt --log-level DEBUG
```

## Future Enhancements

- **Batch API Calls**: Process multiple cases in single API request
- **Confidence Scores**: Add confidence levels to categorizations
- **Custom Categories**: Allow user-defined error categories
- **Model Comparison**: Compare results across different GPT models
- **Interactive Mode**: Manual review and correction of categorizations

## Support

For issues or questions about the GPT categorization functionality:

1. Check the logs for detailed error information
2. Verify your OpenAI API key and quota
3. Review the error case data format
4. Test with a small subset of cases first

## License

This functionality is part of the ClinTrialFinder project and follows the same licensing terms.
