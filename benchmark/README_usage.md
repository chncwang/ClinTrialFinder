# Process Data Label Errors Script

This script processes categorized error cases and handles data label errors. It can now also merge new processed records with existing TSV files.

## Usage

### Basic Usage (without existing TSV)
```bash
python process_data_label_errors.py input.json -o output.tsv
```

### With Existing TSV File (merge mode)
```bash
python process_data_label_errors.py input.json -o output.tsv -e existing_records.tsv
```

## Arguments

- `input_file`: Input JSON file path (required)
- `-o, --output`: Output TSV file path (default: corrected_data_label_errors.tsv)
- `-e, --existing`: Existing TSV file to merge with new processed records (optional)

## Merge Logic

The script implements the following logic:

- **X** = existing records (input tuple set from the optional TSV file)
- **Y** = result of the current process script (new processed records)
- **Z** = final output set where:
  - When X is absent: **Z = Y**
  - When X is present: **Z = {z | z belongs to Y OR (z's query_id and corpus_id combination is not present in Y but present in X)}**

This means:
1. **Existing records cannot be modified** by new processing
2. **Existing records won't be lost** if they don't appear in new processing
3. **New records can overwrite** existing ones with the same query_id + corpus_id combination

## Example

### Input JSON (input.json)
```json
[
  {
    "gpt_categorization": "data_label_error",
    "case_type": "false_negative",
    "patient_id": "patient1",
    "trial_id": "trial1"
  }
]
```

### Existing TSV (existing_records.tsv)
```tsv
query-id	corpus-id	score
patient2	trial2	1
patient3	trial3	2
```

### Output TSV (output.tsv)
```tsv
query-id	corpus-id	score
patient1	trial1	1
patient2	trial2	1
patient3	trial3	2
```

**Explanation**:
- `patient1->trial1` is added from new processing (score=1 for false_negative)
- `patient2->trial2` and `patient3->trial3` are preserved from existing records
- Total: 3 records (1 new + 2 existing)

## Score Mapping

- **false_negative** → score = 1 (true negative)
- **false_positive** → score = 2 (true positive)
