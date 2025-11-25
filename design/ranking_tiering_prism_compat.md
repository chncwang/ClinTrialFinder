# Design: PRISM-Style Tiered Criteria + Weighted Ranking

**Author:** Claude Code
**Date:** 2025-11-25
**Status:** Draft

---

## Executive Summary

This document evaluates whether and how to incorporate a PRISM/OncoLLM-style tiered criteria + weighted ranking mechanism into ClinTrialFinder. After analyzing the current architecture, we present three options with trade-offs, complexity estimates, and a recommended path forward.

**Recommendation:** Start with **Option A** (analytical/logging only) to gather data and validate the approach, then consider **Option B** (hybrid ranking) based on findings. Limit initial scope to NPC/oncology trials.

---

## 1. Current Ranking Behavior

### 1.1 Architecture Overview

ClinTrialFinder uses a three-stage pipeline:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    FILTERING    │ ──▶ │    ANALYSIS     │ ──▶ │    RANKING      │
│  (Title + E/I)  │     │  (Drug + Rec)   │     │  (Quicksort)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 1.2 Current Ranking Logic

**File:** `scripts/rank_trials.py`
**Algorithm:** Quicksort with pairwise GPT comparisons

The ranking is **purely comparative** with no explicit numerical scores:

1. Trials are shuffled (seed=42 for reproducibility)
2. Quicksort partitions trials by comparing pairs via `compare_trials()`
3. GPT-4.1 decides which trial is "better" for the patient
4. Comparison considers:
   - Patient clinical record and disease
   - Trial design, phases, and arms
   - Drug effectiveness evidence (from Perplexity analysis)
   - Recommendation level and reasoning
   - Oncology-specific guidance (Phase 2 vs Phase 3 based on treatment history)

**Key insight:** No scalar scores are computed - the ranking is entirely ordinal via pairwise comparisons.

### 1.3 Criterion-Level Decisions

**File:** `base/trial_expert.py`
**Data Structure:**

```python
@dataclass
class CriterionEvaluation:
    criterion: str      # The criterion text
    reason: str         # Explanation for evaluation
    eligibility: float  # Probability 0.0-1.0
```

During filtering, each inclusion/exclusion criterion is evaluated individually and assigned an eligibility probability. However:

- These probabilities are used for **pass/fail decisions** only
- They are **not stored** in the final trial output
- They are **not used** in ranking (only filtering)
- All criteria are weighted equally

### 1.4 What's Missing for PRISM Compatibility

| PRISM Feature | ClinTrialFinder Status |
|---------------|----------------------|
| Per-criterion Yes/No/N/A labels | Partial - has probabilities, but not stored |
| Tier assignments (T1-T4) | Missing |
| Tier-weighted scoring | Missing |
| Aggregate scalar score | Missing - uses pairwise comparison |
| Patient → Trial ranking | Uses ordinal comparison |
| Trial → Patient ranking | Not implemented |

---

## 2. Options for Adding Tiered Criteria Mechanism

### Option A: Analytical Only (Report Without Changing Ranking)

**Goal:** Compute and log PRISM-style tiered scores alongside existing ranking, without affecting current behavior.

#### Implementation

1. **Store criterion evaluations** in trial output:
   - Modify `evaluate_trial()` to return per-criterion evaluations
   - Add `criterion_evaluations: List[CriterionEvaluation]` to trial dict

2. **Add tier inference** (LLM-based):
   - Create prompt to classify each criterion into T1-T4:
     - T1: Safety-critical (life-threatening conditions, contraindications)
     - T2: Core eligibility (disease type, stage, prior treatments)
     - T3: Important but flexible (lab values, performance status)
     - T4: Administrative/logistical (location, consent, age)

3. **Compute weighted tier score**:
   ```python
   def compute_tiered_score(evaluations: List[CriterionEvaluation],
                            tiers: Dict[str, int]) -> float:
       weights = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}
       tier_scores = defaultdict(list)

       for eval in evaluations:
           tier = tiers.get(eval.criterion, 3)  # Default to T3
           # Convert probability to PRISM-style label
           if eval.eligibility >= 0.8:
               score = 1.0  # Yes
           elif eval.eligibility <= 0.2:
               score = -2.0 if tier <= 2 else -1.0  # No (weighted by tier)
           else:
               score = 0.3  # N/A (weak positive)
           tier_scores[tier].append(score)

       # Weighted average
       total = sum(weights[t] * mean(scores)
                   for t, scores in tier_scores.items() if scores)
       return total
   ```

4. **Log scores** for offline analysis:
   - Add to ranked trial output: `tiered_score`, `tier_breakdown`
   - Enable correlation analysis: tiered_score vs. actual rank position

#### Pros
- **Zero risk** to existing ranking behavior
- Enables **validation** before committing to changes
- Provides **data** for future decisions
- Can be run on historical logs

#### Cons
- Adds GPT calls for tier classification (cost)
- Extra logging complexity
- No immediate user-facing improvement

#### Complexity: **Medium (M)**
- 2-3 days of implementation
- Requires schema changes for storing criterion evaluations
- Needs prompt engineering for tier classification

---

### Option B: Hybrid Ranking (Combine Current + Tiered Score)

**Goal:** Use tiered score as a pre-filter or tie-breaker in the existing quicksort ranking.

#### Implementation Approaches

**B1. Pre-filtering by tiered score:**
```python
def rank_trials_hybrid(trials, clinical_record, disease, gpt_client):
    # Compute tiered scores for all trials
    scored_trials = [(t, compute_tiered_score(t)) for t in trials]

    # Bucket by score range (top/middle/bottom)
    top = [t for t, s in scored_trials if s >= 0.5]
    middle = [t for t, s in scored_trials if -0.5 < s < 0.5]
    bottom = [t for t, s in scored_trials if s <= -0.5]

    # Quicksort within each bucket
    ranked_top = quicksort_with_comparison(top, ...)
    ranked_middle = quicksort_with_comparison(middle, ...)
    ranked_bottom = quicksort_with_comparison(bottom, ...)

    return ranked_top + ranked_middle + ranked_bottom
```

**B2. Tiered score as tie-breaker in comparison:**
```python
def compare_trials_hybrid(trial1, trial2, ...):
    # Existing GPT comparison
    better, reason, cost = compare_trials(trial1, trial2, ...)

    # If GPT says "neither", use tiered score
    if better_trial_id == "neither":
        score1 = compute_tiered_score(trial1)
        score2 = compute_tiered_score(trial2)
        return trial1 if score1 > score2 else trial2

    return better
```

**B3. Include tiered score in comparison prompt:**
```python
comparison_prompt = f"""
...existing prompt...

<tiered_scores>
Trial 1 ({trial1.nct_id}): {trial1.tiered_score}
  - T1 criteria: {trial1.tier_breakdown['T1']}
  - T2 criteria: {trial1.tier_breakdown['T2']}
Trial 2 ({trial2.nct_id}): {trial2.tiered_score}
  ...
</tiered_scores>

Consider these tiered scores alongside other factors.
"""
```

#### Pros
- Incorporates clinical importance weighting
- Could improve ranking quality for edge cases
- Maintains human-readable GPT reasoning

#### Cons
- **Changes production behavior** - needs careful testing
- Tiered scores may conflict with GPT's holistic judgment
- Requires benchmark validation before deployment
- Risk of regression if tier classification is noisy

#### Complexity: **Large (L)**
- 1-2 weeks including testing
- Requires new benchmark metrics
- Needs A/B testing framework or offline evaluation

---

### Option C: Research/Prototype Only (Offline Benchmarking)

**Goal:** Build a standalone script for research experiments without touching production code.

#### Implementation

1. **Create `benchmark/tiered_ranking_experiment.py`:**
   - Load existing filtered/analyzed trials
   - Compute tiered scores using cached criterion evaluations
   - Compare rankings: current vs. tiered-only vs. hybrid

2. **Use existing benchmark infrastructure:**
   - Leverage `benchmark_filtering_performance.py` for evaluation
   - Add metrics: rank correlation, top-K overlap, clinician agreement

3. **Output comparison report:**
   ```
   Experiment: Tiered vs. Current Ranking
   Dataset: NPC trials (n=47)

   Rank Correlation (Spearman): 0.78
   Top-5 Overlap: 4/5 (80%)
   Top-10 Overlap: 7/10 (70%)

   Disagreements:
   - NCT001: Current=3, Tiered=8 (Reason: T1 criterion borderline)
   - NCT042: Current=12, Tiered=5 (Reason: Strong T2 match)
   ```

#### Pros
- **Zero production risk**
- Fast iteration on research ideas
- Can use existing data without new GPT calls
- Informs decision on Option A/B

#### Cons
- No user-facing improvement
- Requires manual interpretation of results
- Limited to datasets with stored criterion evaluations

#### Complexity: **Small (S)**
- 1-2 days
- Uses existing data and infrastructure
- No production code changes

---

## 3. Data Requirements

### 3.1 Where to Get Tier Assignments

| Approach | Pros | Cons |
|----------|------|------|
| **LLM-inferred** | Automatic, no manual work | Cost, potential inconsistency |
| **Rule-based** | Fast, deterministic | Limited coverage, brittle |
| **Manual annotation** | High quality | Labor-intensive, doesn't scale |
| **Hybrid** | Best of both | Complexity |

**Recommendation:** Start with LLM-inferred tiers, validate on small sample, iterate.

### 3.2 Prompt for Tier Classification

```
You are classifying clinical trial eligibility criteria by clinical importance.

Tiers:
- T1 (Critical): Safety-critical conditions that could endanger the patient
  Examples: life-threatening allergies, contraindicated medications, organ failure

- T2 (Core): Essential eligibility defining the target population
  Examples: cancer type/stage, specific mutations, prior treatment lines

- T3 (Important): Meaningful but with some flexibility
  Examples: lab values (within reasonable ranges), ECOG status, washout periods

- T4 (Administrative): Logistical requirements
  Examples: age limits, geographic location, ability to consent, travel requirements

Criterion: "{criterion}"
Trial context: "{trial_title}"

Respond with JSON: {"tier": 1-4, "reason": "brief explanation"}
```

### 3.3 Storing Criterion Evaluations

Currently, criterion evaluations are computed during filtering but **not persisted**. To enable tiered scoring:

**Option 1: Add to trial JSON output**
```json
{
  "identification": {...},
  "criterion_evaluations": [
    {
      "criterion": "Patient must have histologically confirmed NPC",
      "eligibility": 0.95,
      "reason": "Patient has confirmed NPC diagnosis",
      "tier": 2
    },
    ...
  ]
}
```

**Option 2: Separate log file**
- Keep trial JSON clean
- Store evaluations in `results/criterion_evaluations_{timestamp}.json`
- Link by NCT ID

**Recommendation:** Option 1 for simplicity, with optional flag to exclude in output.

---

## 4. Pros/Cons Summary

| Aspect | Option A | Option B | Option C |
|--------|----------|----------|----------|
| Production risk | None | High | None |
| Implementation effort | Medium | Large | Small |
| User-facing value | None (data gathering) | Potential improvement | None |
| Validation data | Yes | Requires validation | Uses existing |
| Recommended first step | **Yes** | After A validation | Yes (parallel) |

---

## 5. Recommendation

### Recommended Path

1. **Start with Option C** (1-2 days):
   - Build offline experiment script
   - Use existing NPC trial logs
   - Measure correlation with current ranking
   - Quick validation of the approach

2. **Proceed to Option A if promising** (3-5 days):
   - Implement criterion evaluation storage
   - Add tier inference via LLM
   - Compute and log tiered scores
   - Gather data over several runs

3. **Consider Option B only after**:
   - Option A shows consistent correlation improvements
   - Benchmark validation passes
   - Clinical review of tier assignments

### Scope Limitation

**Limit to NPC/oncology initially:**
- Oncology criteria have clearer tier distinctions
- Existing test data available
- Personal investment in NPC use case

**Do NOT generalize yet:**
- Different disease areas may need different tier schemas
- Validation burden increases with scope

### Decision Criteria for Moving Forward

| Metric | Threshold for Option B |
|--------|----------------------|
| Rank correlation (current vs. tiered) | > 0.7 |
| Top-5 overlap | > 80% |
| Tier classification accuracy (manual review) | > 85% |
| Clinical agreement (spot check) | Positive feedback |

---

## 6. Task Checklist

### Safe to Do Now (S)

| Task | Size | Description |
|------|------|-------------|
| Create `benchmark/tiered_ranking_experiment.py` | S | Offline experiment script using existing data |
| Draft tier classification prompt | S | Prompt engineering for T1-T4 assignment |
| Analyze existing logs for criterion data | S | Check what's already captured in logs |
| Define evaluation metrics | S | Spearman correlation, top-K overlap, etc. |

### Need More Data/Annotation (M)

| Task | Size | Description |
|------|------|-------------|
| Modify `evaluate_trial()` to return criterion evaluations | M | Schema change to persist evaluations |
| Add `criterion_evaluations` to trial output schema | M | JSON structure change |
| Implement tier inference function | M | LLM call with caching |
| Compute and log tiered scores | M | Scoring function + logging |
| Manual review of tier assignments (sample) | M | 50-100 criteria spot check |

### Research/Experiment Only (L)

| Task | Size | Description |
|------|------|-------------|
| Build hybrid ranking prototype | L | Option B1, B2, or B3 implementation |
| A/B testing framework | L | Compare rankings side-by-side |
| Full benchmark with tiered vs. current | L | Statistical significance testing |
| Clinical validation of hybrid ranking | L | Domain expert review |

---

## 7. Appendix: PRISM/OncoLLM Reference

From the PRISM paper, the Weighted Tier scoring formula:

```
Score = Σ (w_t × Σ s_c for c in tier_t) / Σ w_t

Where:
- w_t = weight for tier t (T1=0.4, T2=0.3, T3=0.2, T4=0.1)
- s_c = criterion score:
  - Yes: +1
  - No: -2 (T1/T2) or -1 (T3/T4)
  - N/A: +0.3
```

The key insight is that **violating high-tier criteria is heavily penalized**, while missing information (N/A) is treated as a weak positive to avoid over-penalizing incomplete records.

---

## 8. Open Questions

1. **Should tiered scores influence filtering or only ranking?**
   - Current: Filtering uses hard pass/fail
   - PRISM: Uses tiered score for ranking only

2. **How to handle criteria that span multiple tiers?**
   - Example: "ECOG 0-2 with no active infections"
   - Split into sub-criteria? Assign to highest tier?

3. **Should tier assignments be trial-specific or universal?**
   - Example: "Prior checkpoint inhibitor" might be T2 for some trials, T3 for others

4. **Cost implications of tier classification?**
   - Estimate: ~$0.01-0.02 per criterion (GPT-4-mini)
   - For 20 criteria × 50 trials = ~$10-20 per run

---

*Document generated by Claude Code on 2025-11-25*
