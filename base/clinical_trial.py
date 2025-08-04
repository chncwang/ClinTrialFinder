import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from base.gpt_client import GPTClient

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class Identification:
    nct_id: str
    url: str
    brief_title: str
    official_title: str
    acronym: Optional[str]
    org_study_id: str


@dataclass
class Status:
    overall_status: str
    start_date: Optional[datetime]
    completion_date: Optional[datetime]
    primary_completion_date: Optional[datetime]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Status":
        def parse_date(date_str: Optional[str]) -> Optional[datetime]:
            if not date_str:
                return None
            try:
                return datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                return None

        return cls(
            overall_status=data["overall_status"],
            start_date=parse_date(data.get("start_date")),
            completion_date=parse_date(data.get("completion_date")),
            primary_completion_date=parse_date(data.get("primary_completion_date")),
        )


@dataclass
class Description:
    brief_summary: str
    detailed_description: str
    conditions: Optional[List[str]]
    keywords: Optional[List[str]]


class Phase(Enum):
    PHASE1 = 1
    PHASE2 = 2
    PHASE3 = 3
    PHASE4 = 4


@dataclass
class Design:
    study_type: str
    phases: List[int]
    enrollment: int
    arms: List[Dict[str, Any]]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Design":
        raw_phases = data.get("phases", [])
        parsed_phases: List[int] = []

        for phase in raw_phases:
            try:
                if isinstance(phase, int):
                    phase_num = phase
                else:
                    phase_str = phase.lower().replace(" ", "")
                    # Extract any digits from the phase string
                    phase_digits = "".join(c for c in phase_str if c.isdigit())
                    if not phase_digits:
                        continue
                    phase_num = int(phase_digits)

                if 1 <= phase_num <= 4:
                    parsed_phases.append(phase_num)
                else:
                    raise ValueError(f"Invalid phase number: {phase_num}")
            except (ValueError, AttributeError):
                raise ValueError(
                    f"Invalid phase format: {phase}. Expected phase number between 1-4"
                )

        return cls(
            study_type=data["study_type"],
            phases=parsed_phases,
            enrollment=data["enrollment"],
            arms=data["arms"],
        )


@dataclass
class Eligibility:
    criteria: str
    gender: str
    minimum_age: str
    maximum_age: str
    healthy_volunteers: bool


@dataclass
class Location:
    facility: Optional[str]
    city: Optional[str]
    state: Optional[str]
    country: Optional[str]
    status: Optional[str]


@dataclass
class ContactsLocations:
    locations: List[Location]


@dataclass
class Sponsor:
    lead_sponsor: str
    collaborators: List[str]


class ClinicalTrial:
    def __init__(self, trial_data: Dict[str, Any]):
        self.identification = Identification(**trial_data["identification"])
        self.status = Status.from_dict(trial_data["status"])
        self.description = Description(**trial_data["description"])
        self.design = Design.from_dict(trial_data["design"])
        self.eligibility = Eligibility(**trial_data["eligibility"])

        locations_data = trial_data["contacts_locations"]["locations"]
        self.contacts_locations = ContactsLocations(
            locations=[Location(**loc) for loc in locations_data]
        )

        self.sponsor = Sponsor(**trial_data["sponsor"])

        # Handle additional fields from analyzed trials
        self.recommendation_level = trial_data.get("recommendation_level")
        self.analysis_reason = trial_data.get("reason")
        self.drug_analysis = trial_data.get("drug_analysis")

    @property
    def is_recruiting(self) -> bool:
        return self.status.overall_status == "RECRUITING"

    @property
    def study_duration_days(self) -> Optional[int]:
        if self.status.start_date and self.status.completion_date:
            return (self.status.completion_date - self.status.start_date).days
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the trial object back to a dictionary format."""
        base_dict = {
            "identification": {
                "nct_id": self.identification.nct_id,
                "brief_title": self.identification.brief_title,
                "official_title": self.identification.official_title,
                "acronym": self.identification.acronym,
                "org_study_id": self.identification.org_study_id,
                "url": self.identification.url,
            },
            "status": {
                "overall_status": self.status.overall_status,
                "start_date": (
                    self.status.start_date.strftime("%Y-%m-%d")
                    if self.status.start_date
                    else None
                ),
                "completion_date": (
                    self.status.completion_date.strftime("%Y-%m-%d")
                    if self.status.completion_date
                    else None
                ),
                "primary_completion_date": (
                    self.status.primary_completion_date.strftime("%Y-%m-%d")
                    if self.status.primary_completion_date
                    else None
                ),
            },
            "description": {
                "brief_summary": self.description.brief_summary,
                "detailed_description": self.description.detailed_description,
                "conditions": self.description.conditions,
                "keywords": self.description.keywords,
            },
            "design": {
                "study_type": self.design.study_type,
                "phases": self.design.phases,
                "enrollment": self.design.enrollment,
                "arms": self.design.arms,
            },
            "eligibility": {
                "criteria": self.eligibility.criteria,
                "gender": self.eligibility.gender,
                "minimum_age": self.eligibility.minimum_age,
                "maximum_age": self.eligibility.maximum_age,
                "healthy_volunteers": self.eligibility.healthy_volunteers,
            },
            "contacts_locations": {
                "locations": [
                    {
                        "facility": loc.facility,
                        "city": loc.city,
                        "state": loc.state,
                        "country": loc.country,
                        "status": loc.status,
                    }
                    for loc in self.contacts_locations.locations
                ]
            },
            "sponsor": {
                "lead_sponsor": self.sponsor.lead_sponsor,
                "collaborators": self.sponsor.collaborators,
            },
        }

        # Add analyzed trial fields if they exist
        if (
            hasattr(self, "recommendation_level")
            and self.recommendation_level is not None
        ):
            base_dict["recommendation_level"] = self.recommendation_level
        if hasattr(self, "analysis_reason") and self.analysis_reason is not None:
            base_dict["reason"] = self.analysis_reason
        if hasattr(self, "drug_analysis") and self.drug_analysis is not None:
            base_dict["drug_analysis"] = self.drug_analysis

        return base_dict

    def __str__(self) -> str:
        """Provide a user-friendly string representation of the ClinicalTrial object."""
        return (
            f"ClinicalTrial(nct_id={self.identification.nct_id}, "
            f"brief_title={self.identification.brief_title}, "
            f"overall_status={self.status.overall_status})"
        )

    def __repr__(self) -> str:
        """Provide a detailed string representation of the ClinicalTrial object."""
        return (
            f"ClinicalTrial(nct_id={self.identification.nct_id}, "
            f"url={self.identification.url}, "
            f"brief_title={self.identification.brief_title}, "
            f"official_title={self.identification.official_title}, "
            f"acronym={self.identification.acronym}, "
            f"org_study_id={self.identification.org_study_id}, "
            f"overall_status={self.status.overall_status}, "
            f"start_date={self.status.start_date}, "
            f"completion_date={self.status.completion_date}, "
            f"primary_completion_date={self.status.primary_completion_date}, "
            f"brief_summary={self.description.brief_summary}, "
            f"detailed_description={self.description.detailed_description}, "
            f"conditions={self.description.conditions}, "
            f"keywords={self.description.keywords}, "
            f"study_type={self.design.study_type}, "
            f"phases={self.design.phases}, "
            f"enrollment={self.design.enrollment}, "
            f"arms={self.design.arms}, "
            f"criteria={self.eligibility.criteria}, "
            f"gender={self.eligibility.gender}, "
            f"minimum_age={self.eligibility.minimum_age}, "
            f"maximum_age={self.eligibility.maximum_age}, "
            f"healthy_volunteers={self.eligibility.healthy_volunteers}, "
            f"locations={self.contacts_locations.locations}, "
            f"lead_sponsor={self.sponsor.lead_sponsor}, "
            f"collaborators={self.sponsor.collaborators})"
        )

    def get_novel_drugs_from_arms(
        self, gpt_client: "GPTClient"
    ) -> Tuple[List[str], float]:
        """
        Extract novel drug names from the interventions in all arms using GPT.
        Args:
            gpt_client: Initialized GPTClient instance for making API calls
        Returns:
            Tuple containing:
            - List of novel drug names found in the arms
            - Cost of the GPT API call
        """
        # Build arms information for GPT analysis
        arms_info: List[str] = []
        for i, arm in enumerate(self.design.arms, 1):
            arm_name = arm.get('name', f'Arm {i}')
            arm_type = arm.get('type', 'Unknown')
            arm_description = arm.get('description', '')
            interventions = arm.get('interventions', [])

            arm_str = f"Arm {i}: {arm_name} ({arm_type})"
            if arm_description:
                arm_str += f" - {arm_description}"
            if interventions:
                if isinstance(interventions, list):
                    intervention_strs: List[str] = [str(x) for x in interventions if x is not None]  # type: ignore
                    arm_str += f" - Interventions: {', '.join(intervention_strs)}"
                else:
                    arm_str += f" - Interventions: {interventions}"
            arms_info.append(arm_str)

        arms_text = "\n".join(arms_info)
        logger.info(f"get_novel_drugs_from_arms: arms_text: {arms_text}")

        prompt = f"""Analyze these clinical trial arms and extract any novel drug names mentioned in the interventions:

Rules for extraction:
1. Only include drugs that appear to be novel or experimental
2. Exclude common/established drugs
3. Return drug names in their exact form from the interventions
4. If no novel drugs are mentioned, return an empty list
5. Include both code names (e.g., BMS-936558) and generic names if present

Return a JSON object with a single key "drug_names" containing an array of strings.
Example: {{"drug_names": ["BMS-936558", "nivolumab"]}}

Arms Information:
{arms_text}
"""

        system_role = "You are a pharmaceutical expert focused on identifying novel and experimental drugs in clinical trial interventions."

        try:
            response, cost = gpt_client.call_with_retry(
                prompt,
                system_role,
                temperature=0.1,
                response_format={"type": "json_object"},
                validate_json=True,
            )

            if isinstance(response, dict):
                drug_names: List[str] = response.get("drug_names", []) or []
            else:
                drug_names: List[str] = []
            return drug_names, cost

        except Exception as e:
            logger.error(f"Failed to extract novel drugs from arms: {str(e)}")
            return [], 0.0

    def get_novel_drugs_from_title(
        self, gpt_client: "GPTClient"
    ) -> Tuple[List[str], float]:
        """
        Extract novel drug names mentioned in the trial's brief title using GPT.

        Args:
            gpt_client: Initialized GPTClient instance for making API calls

        Returns:
            Tuple containing:
            - List of novel drug names found in the title
            - Cost of the GPT API call
        """
        prompt = f"""Analyze this clinical trial title and extract any novel drug names mentioned:

Title: {self.identification.brief_title}

Rules for extraction:
1. Only include drugs that appear to be novel or experimental
2. Exclude common/established drugs
3. Return drug names in their exact form from the title
4. If no novel drugs are mentioned, return an empty list
5. Include both code names (e.g., BMS-936558) and generic names if present

Return a JSON object with a single key "drug_names" containing an array of strings.
Example: {{"drug_names": ["BMS-936558", "nivolumab"]}}"""

        system_role = "You are a pharmaceutical expert focused on identifying novel and experimental drugs in clinical trial titles."

        try:
            response, cost = gpt_client.call_with_retry(
                prompt,
                system_role,
                temperature=0.1,
                response_format={"type": "json_object"},
                validate_json=True,
            )

            if isinstance(response, dict):
                drug_names: List[str] = response.get("drug_names", []) or []
            else:
                drug_names: List[str] = []
            return drug_names, cost

        except Exception as e:
            logger.error(f"Failed to extract novel drugs from title: {str(e)}")
            return [], 0.0


class ClinicalTrialsParser:
    def __init__(self, json_data: List[Dict[str, Any]]):
        self.trials = [ClinicalTrial(trial_data) for trial_data in json_data]

    def get_trial_by_nct_id(self, nct_id: str) -> Optional[ClinicalTrial]:
        """Retrieve a specific trial by its NCT ID."""
        for trial in self.trials:
            if trial.identification.nct_id == nct_id:
                return trial
        return None

    def get_recruiting_trials(self) -> List[ClinicalTrial]:
        """Get all trials that are currently recruiting."""
        return [trial for trial in self.trials if trial.is_recruiting]

    def get_trials_by_phase(self, phase: int) -> List[ClinicalTrial]:
        """Get all trials for a specific phase."""
        return [trial for trial in self.trials if phase in trial.design.phases]

    def get_trials_excluding_study_type(
        self, excluded_type: str
    ) -> List[ClinicalTrial]:
        """Get all trials except those matching the specified study type (case-insensitive).

        Args:
            excluded_type: The study type to exclude (e.g., 'Observational', 'Interventional')

        Returns:
            List of trials that don't match the excluded study type
        """
        return [
            trial
            for trial in self.trials
            if trial.design.study_type.lower() != excluded_type.lower()
        ]

    def get_trials_by_recommendation_level(
        self, recommendation_level: str
    ) -> List[ClinicalTrial]:
        """Get all trials matching the specified recommendation level (case-insensitive).

        Args:
            recommendation_level: The recommendation level to filter by (e.g., 'Strongly Recommended',
                                'Recommended', 'Neutral', 'Not Recommended')

        Returns:
            List of trials that match the recommendation level
        """
        # For "Recommended", match substring but exclude "Not Recommended"
        if recommendation_level.lower() == "recommended":
            return [
                trial
                for trial in self.trials
                if (
                    hasattr(trial, "recommendation_level")
                    and trial.recommendation_level
                    and "recommended" in trial.recommendation_level.lower()
                    and "not recommended" not in trial.recommendation_level.lower()
                    and "not_recommended" not in trial.recommendation_level.lower()
                )
            ]

        # For all other cases, use substring matching
        return [
            trial
            for trial in self.trials
            if (
                hasattr(trial, "recommendation_level")
                and trial.recommendation_level
                and recommendation_level.lower() in trial.recommendation_level.lower()
            )
        ]

    def get_trials_with_novel_drug_analysis(self) -> List[ClinicalTrial]:
        """Get all trials that have non-empty drug analysis results.

        Returns:
            List of trials that have drug analysis results (i.e., novel drugs were found and analyzed)
        """
        return [
            trial
            for trial in self.trials
            if hasattr(trial, "drug_analysis")
            and trial.drug_analysis is not None
            and trial.drug_analysis  # Check if dict is non-empty
        ]
