# TrialGPT LLM Aggregation Prompts

This document compares TrialGPT's original LLM aggregation approach with ClinTrialFinder's implementation.

Sources:
- Official TrialGPT repo: `/Users/chncwang/Projects/TrialGPT`
- Reference: https://github.com/ncbi-nlp/TrialGPT
- Matching code: `trialgpt_matching/TrialGPT.py`
- Ranking/aggregation code: `trialgpt_ranking/TrialGPT.py`
- Score combination: `trialgpt_ranking/rank_results.py`

## TrialGPT's Two-Stage Approach

### Stage 1: Matching (Batch Criterion Evaluation)

**Source:** `trialgpt_matching/TrialGPT.py` (lines 59-89)

Evaluates ALL criteria in one prompt with categorical labels:
- **Labels for inclusion:** "not applicable", "not enough information", "included", "not included"
- **Labels for exclusion:** "not applicable", "not enough information", "excluded", "not excluded"
- Outputs JSON with 3 elements per criterion:
  - Element 1: Brief reasoning process
  - Element 2: Relevant sentence IDs from patient note
  - Element 3: Eligibility label

**System Prompt (Inclusion):**
```
You are a helpful assistant for clinical trial recruitment. Your task is to compare a given patient note and the inclusion criteria of a clinical trial to determine the patient's eligibility at the criterion level.

The factors that allow someone to participate in a clinical study are called inclusion criteria. They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.

You should check the inclusion criteria one-by-one, and output the following three elements for each criterion:
	Element 1. For each inclusion criterion, briefly generate your reasoning process: First, judge whether the criterion is not applicable (not very common), where the patient does not meet the premise of the criterion. Then, check if the patient note contains direct evidence. If so, judge whether the patient meets or does not meet the criterion. If there is no direct evidence, try to infer from existing evidence, and answer one question: If the criterion is true, is it possible that a good patient note will miss such information? If impossible, then you can assume that the criterion is not true. Otherwise, there is not enough information.
	Element 2. If there is relevant information, you must generate a list of relevant sentence IDs in the patient note. If there is no relevant information, you must annotate an empty list.
	Element 3. Classify the patient eligibility for this specific inclusion criterion: the label must be chosen from {"not applicable", "not enough information", "included", "not included"}. "not applicable" should only be used for criteria that are not applicable to the patient. "not enough information" should be used where the patient note does not contain sufficient information for making the classification. Try to use as less "not enough information" as possible because if the note does not mention a medically important fact, you can assume that the fact is not true for the patient. "included" denotes that the patient meets the inclusion criterion, while "not included" means the reverse.

You should output only a JSON dict exactly formatted as: dict{str(criterion_number): list[str(element_1_brief_reasoning), list[int(element_2_sentence_id)], str(element_3_eligibility_label)]}.
```

**User Prompt:**
```
Here is the patient note, each sentence is led by a sentence_id:
{patient_with_sentence_ids}

Here is the clinical trial:
{trial_info_with_criteria}

Plain JSON output:
```

**Temperature:** 0

### Stage 2: Aggregation (Holistic Scoring)

**Source:** `trialgpt_ranking/TrialGPT.py` (lines 66-96, function `convert_pred_to_prompt`)

**System Prompt:**
```
You are a helpful assistant for clinical trial recruitment. You will be given a patient note, a clinical trial, and the patient eligibility predictions for each criterion.
Your task is to output two scores, a relevance score (R) and an eligibility score (E), between the patient and the clinical trial.
First explain the consideration for determining patient-trial relevance. Predict the relevance score R (0~100), which represents the overall relevance between the patient and the clinical trial. R=0 denotes the patient is totally irrelevant to the clinical trial, and R=100 denotes the patient is exactly relevant to the clinical trial.
Then explain the consideration for determining patient-trial eligibility. Predict the eligibility score E (-R~R), which represents the patient's eligibility to the clinical trial. Note that -R <= E <= R (the absolute value of eligibility cannot be higher than the relevance), where E=-R denotes that the patient is ineligible (not included by any inclusion criteria, or excluded by all exclusion criteria), E=R denotes that the patient is eligible (included by all inclusion criteria, and not excluded by any exclusion criteria), E=0 denotes the patient is neutral (i.e., no relevant information for all inclusion and exclusion criteria).
Please output a JSON dict formatted as Dict{"relevance_explanation": Str, "relevance_score_R": Float, "eligibility_explanation": Str, "eligibility_score_E": Float}.
```

**User Prompt:**
```
Here is the patient note:
{patient_note}

Here is the clinical trial description:
{trial_description}

Here are the criterion-level eligibility prediction:
{formatted_matching_results}

Plain JSON output:
```

**Trial Description Format:**
```
Title: {brief_title}
Target conditions: {diseases_list}
Summary: {brief_summary}
```

**Criterion-level Prediction Format:**
```
inclusion criterion 0: {criterion_text}
	Patient relevance: {reasoning_from_element_1}
	Evident sentences: {sentence_ids_from_element_2}
	Patient eligibility: {label_from_element_3}
```

**Temperature:** 0

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

**Source:** `trialgpt_ranking/rank_results.py` (lines 12-78, 106)

```python
eps = 1e-9

def get_matching_score(matching):
    # Count inclusion labels
    included = 0
    not_inc = 0
    no_info_inc = 0

    for criteria, info in matching["inclusion"].items():
        if len(info) != 3:
            continue
        if info[2] == "included":
            included += 1
        elif info[2] == "not included":
            not_inc += 1
        elif info[2] == "not enough information":
            no_info_inc += 1
        # "not applicable" is not counted

    # Count exclusion labels
    excluded = 0
    for criteria, info in matching["exclusion"].items():
        if len(info) != 3:
            continue
        if info[2] == "excluded":
            excluded += 1

    # Compute matching score
    score = included / (included + not_inc + no_info_inc + eps)
    if not_inc > 0:
        score -= 1
    if excluded > 0:
        score -= 1

    return score

def get_agg_score(assessment):
    rel_score = float(assessment["relevance_score_R"])
    eli_score = float(assessment["eligibility_score_E"])
    score = (rel_score + eli_score) / 100
    return score

# Final score (no normalization in original TrialGPT)
trial_score = matching_score + agg_score
```

**Score Range:**
- `matching_score` range: roughly [-2, 1]
  - Best case: all included, no failures → 1.0
  - Worst case: has not_inc + has excluded → -2.0
- `agg_score` range: [0, 2]
  - R range: [0, 100]
  - E range: [-R, R], so [-100, 100]
  - (R + E) / 100 range: [0, 2]
- `trial_score` range: roughly [-2, 3]

**Note:** Original TrialGPT does NOT normalize to [0, 1]. Scores can be negative.

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

## Key Insights from Original TrialGPT Repo

1. **No Normalization:** TrialGPT does NOT normalize final scores to [0, 1]. Trial scores can be negative.

2. **Temperature = 0:** Both matching and aggregation stages use `temperature=0` for deterministic responses.

3. **Separate Prompts:** Inclusion and exclusion criteria are evaluated in **separate API calls** (not in one batch).

4. **Sentence IDs:** Patient notes are pre-processed to add sentence IDs (e.g., "0. Patient is a 65-year-old male..."), which are referenced in matching results.

5. **Trial Summary:** Aggregation prompt includes trial title, target conditions, and brief summary (NOT full criteria).

6. **Matching Format:** Element 2 (sentence IDs) can be empty list `[]` if no relevant evidence found.

7. **Original Model:** TrialGPT paper used GPT-4 (not GPT-4.1 mini or GPT-4o).

## Recommendation

**For the paper implementation:**
- We should use TrialGPT's exact prompt format to ensure fair comparison
- This means implementing the two-score system (R + E) with feature combination
- Current CTF implementation should be updated to match TrialGPT's aggregation prompt
- Keep scores unnormalized (can be negative) to match original TrialGPT behavior

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
