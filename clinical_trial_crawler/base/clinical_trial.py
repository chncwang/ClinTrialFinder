from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


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
        parsed_phases = []

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
        return {
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
