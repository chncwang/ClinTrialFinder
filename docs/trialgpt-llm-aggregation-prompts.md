# TrialGPT LLM Aggregation Prompts

This document compares TrialGPT's original LLM aggregation approach with ClinTrialFinder's implementation.

## TrialGPT's Two-Stage Approach

Reference: https://github.com/ncbi-nlp/TrialGPT

### Stage 1: Matching (Batch Criterion Evaluation)
- Evaluates ALL criteria in one prompt with categorical labels
- Labels: "not applicable", "not enough information", "included/excluded", "not included/not excluded"
- Outputs JSON with reasoning, relevant sentence IDs, and labels for each criterion

### Stage 2: Aggregation (Holistic Scoring)

**System Prompt:**
```
You are a helpful assistant for clinical trial recruitment. You will be given a patient note, a clinical trial, and the patient eligibility predictions for each criterion.

Your task is to output two scores, a relevance score (R) and an eligibility score (E), between the patient and the clinical trial.

First explain the consideration for determining patient-trial relevance. Predict the relevance score R (0~100), which represents the overall relevance between the patient and the clinical trial. R=0 denotes the patient is totally irrelevant to the clinical trial, and R=100 denotes the patient is exactly relevant to the clinical trial.

Then explain the consideration for determining patient-trial eligibility. Predict the eligibility score E (-R~R), which represents the patient's eligibility to the clinical trial. Note that -R <= E <= R (the absolute value of eligibility cannot be higher than the relevance), where E=-R denotes that the patient is ineligible (not included by any inclusion criteria, or excluded by all exclusion criteria), E=R denotes that the patient is eligible (included by all inclusion criteria, and not excluded by any exclusion criteria), E=0 denotes the patient is neutral (i.e., no relevant information for all inclusion and exclusion criteria).

Please output a JSON dict formatted as Dict{"relevance_explanation": Str, "relevance_score_R": Float, "eligibility_explanation": Str, "eligibility_score_E": Float}.
```

**User Prompt Structure:**
```
Here is the patient note:
{patient_with_sentence_ids}

Here is the clinical trial description:
{trial_summary}

Here are the criterion-level eligibility prediction:
{formatted_matching_results}

Plain JSON output:
```

**Output Format:**
```json
{
  "relevance_explanation": "...",
  "relevance_score_R": 75.0,
  "eligibility_explanation": "...",
  "eligibility_score_E": 50.0
}
```

**Final Score Calculation:**
```python
# Feature combination (matching + aggregation)
matching_score = included/(included+not_inc+no_info+eps) - (1 if not_inc>0 else 0) - (1 if excluded>0 else 0)
agg_score = (R + E) / 100
raw_score = matching_score + agg_score

# Normalize to [0, 1]
normalized_score = (raw_score + 2) / 5  # Maps [-2, 3] → [0, 1]
normalized_score = max(0.0, min(1.0, normalized_score))
```

## ClinTrialFinder's Simplified LLM Aggregation

Our implementation simplifies TrialGPT's approach for better clarity and clinical interpretability:

### Key Differences

1. **Single Score Instead of R+E:**
   - TrialGPT outputs: Relevance (R) + Eligibility (E)
   - CTF outputs: Single eligibility score [0.0-1.0]

2. **No Feature Combination:**
   - TrialGPT combines: matching_score + (R+E)/100
   - CTF uses: LLM aggregation score only

3. **Different Criterion Format:**
   - TrialGPT: Batch evaluation with categorical labels + sentence IDs
   - CTF: Sequential evaluation with probability scores

### ClinTrialFinder System Prompt

```
You are a medical expert evaluating clinical trial eligibility with a holistic, patient-centered approach.
```

### ClinTrialFinder User Prompt

```
You are evaluating a patient's overall eligibility for a clinical trial by holistically reviewing per-criterion evaluation results.

Trial Title: {trial_title}

Patient Conditions: {conditions}

Per-Criterion Evaluation Results:
{formatted_criterion_evaluations}

Instructions:
1. Review all criterion scores holistically
2. Consider whether criterion failures are absolute exclusions or negotiable
3. A score of 0.0 typically means complete failure, 1.0 means complete pass, values in between indicate partial match or uncertainty
4. Consider the overall disease match and trial suitability despite individual criterion failures
5. For patients with comorbidities (e.g., on hemodialysis), don't zero out trials just because of one organ function criterion if the primary disease criteria match well
6. Output a final eligibility score [0.0-1.0] representing overall trial suitability
7. Provide brief reasoning for your score

Output your response as JSON with the following format:
{
    "final_score": <float between 0.0 and 1.0>,
    "reasoning": "<brief explanation of your scoring decision>"
}
```

## Recommendation

**For the paper implementation:**
- We should use TrialGPT's exact prompt format to ensure fair comparison
- This means implementing the two-score system (R + E) with feature combination
- Current CTF implementation should be updated to match TrialGPT's aggregation prompt

**Current Status:**
- CTF has simplified aggregation implemented in `_aggregate_criteria_with_llm()`
- Need to update to match TrialGPT's R+E scoring for paper benchmark

## Implementation TODO

- [ ] Update `_aggregate_criteria_with_llm()` to use TrialGPT's R+E scoring format
- [ ] Implement feature combination: matching_score + agg_score
- [ ] Add matching_score calculation from per-criterion labels
- [ ] Update system prompt to match TrialGPT exactly
- [ ] Test on Patient 17 to verify improved handling
- [ ] Run 20-patient benchmark with both approaches
