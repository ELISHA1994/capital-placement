"""Domain service for profile-job matching business logic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from app.domain.entities.profile import Profile, ExperienceLevel, Skill
from app.domain.value_objects import MatchScore


@dataclass
class JobRequirements:
    """Value object representing job requirements for matching."""
    
    required_skills: List[str]
    preferred_skills: List[str] = None
    minimum_experience_years: Optional[int] = None
    experience_level: Optional[ExperienceLevel] = None
    location_requirements: Optional[str] = None
    education_requirements: Optional[str] = None

    def __post_init__(self):
        if self.preferred_skills is None:
            self.preferred_skills = []


@dataclass
class MatchResult:
    """Result of profile-job matching."""
    
    profile: Profile
    overall_score: MatchScore
    skills_score: MatchScore
    experience_score: MatchScore
    education_score: MatchScore
    location_score: MatchScore
    matching_skills: List[str]
    missing_skills: List[str]
    reasons: List[str]

    @property
    def is_good_match(self) -> bool:
        """Determine if this is considered a good match."""
        return self.overall_score.value >= 0.7

    @property
    def is_excellent_match(self) -> bool:
        """Determine if this is an excellent match."""
        return self.overall_score.value >= 0.9


class IMatchingService(ABC):
    """Domain service interface for profile-job matching."""

    @abstractmethod
    def calculate_match(self, profile: Profile, requirements: JobRequirements) -> MatchResult:
        """Calculate how well a profile matches job requirements."""
        pass

    @abstractmethod
    def rank_profiles(
        self, 
        profiles: List[Profile], 
        requirements: JobRequirements
    ) -> List[MatchResult]:
        """Rank multiple profiles against job requirements."""
        pass

    @abstractmethod
    def find_skill_gaps(
        self, 
        profile: Profile, 
        requirements: JobRequirements
    ) -> List[str]:
        """Identify skills the profile is missing for the job."""
        pass


class MatchingService(IMatchingService):
    """Concrete implementation of profile-job matching service."""

    def calculate_match(self, profile: Profile, requirements: JobRequirements) -> MatchResult:
        """Calculate comprehensive match score between profile and job requirements."""
        skills_score = self._calculate_skills_score(profile, requirements)
        experience_score = self._calculate_experience_score(profile, requirements)
        education_score = self._calculate_education_score(profile, requirements)
        location_score = self._calculate_location_score(profile, requirements)
        
        # Weighted overall score
        overall_score = MatchScore(
            (skills_score.value * 0.5) +
            (experience_score.value * 0.3) +
            (education_score.value * 0.1) +
            (location_score.value * 0.1)
        )
        
        matching_skills = self._find_matching_skills(profile, requirements)
        missing_skills = self._find_missing_skills(profile, requirements)
        reasons = self._generate_match_reasons(
            profile, requirements, skills_score, experience_score
        )
        
        return MatchResult(
            profile=profile,
            overall_score=overall_score,
            skills_score=skills_score,
            experience_score=experience_score,
            education_score=education_score,
            location_score=location_score,
            matching_skills=matching_skills,
            missing_skills=missing_skills,
            reasons=reasons
        )

    def rank_profiles(
        self, 
        profiles: List[Profile], 
        requirements: JobRequirements
    ) -> List[MatchResult]:
        """Rank profiles by match score in descending order."""
        matches = [self.calculate_match(profile, requirements) for profile in profiles]
        return sorted(matches, key=lambda m: m.overall_score.value, reverse=True)

    def find_skill_gaps(
        self, 
        profile: Profile, 
        requirements: JobRequirements
    ) -> List[str]:
        """Find skills missing from profile that are required for job."""
        return self._find_missing_skills(profile, requirements)

    def _calculate_skills_score(
        self, 
        profile: Profile, 
        requirements: JobRequirements
    ) -> MatchScore:
        """Calculate skills match score."""
        if not requirements.required_skills:
            return MatchScore(1.0)
        
        profile_skills = set(skill.lower() for skill in profile.normalized_skills)
        required_skills = set(skill.lower() for skill in requirements.required_skills)
        preferred_skills = set(skill.lower() for skill in requirements.preferred_skills)
        
        # Score based on required skills
        matching_required = len(required_skills.intersection(profile_skills))
        required_score = matching_required / len(required_skills) if required_skills else 1.0
        
        # Bonus for preferred skills
        matching_preferred = len(preferred_skills.intersection(profile_skills))
        preferred_bonus = min(0.2, (matching_preferred / len(preferred_skills)) * 0.2) if preferred_skills else 0.0
        
        final_score = min(1.0, required_score + preferred_bonus)
        return MatchScore(final_score)

    def _calculate_experience_score(
        self, 
        profile: Profile, 
        requirements: JobRequirements
    ) -> MatchScore:
        """Calculate experience match score."""
        # If no experience requirements, give perfect score
        if not requirements.minimum_experience_years and not requirements.experience_level:
            return MatchScore(1.0)
        
        total_experience = profile.profile_data.total_experience_years()
        
        # Score based on years of experience
        years_score = 1.0
        if requirements.minimum_experience_years:
            if total_experience >= requirements.minimum_experience_years:
                years_score = 1.0
            else:
                # Partial credit for partial experience
                years_score = max(0.3, total_experience / requirements.minimum_experience_years)
        
        # Score based on experience level
        level_score = 1.0
        if requirements.experience_level and profile.experience_level:
            level_values = {
                ExperienceLevel.ENTRY: 1,
                ExperienceLevel.JUNIOR: 2,
                ExperienceLevel.MID: 3,
                ExperienceLevel.SENIOR: 4,
                ExperienceLevel.LEAD: 5,
                ExperienceLevel.PRINCIPAL: 6,
                ExperienceLevel.EXECUTIVE: 7
            }
            
            required_level = level_values.get(requirements.experience_level, 3)
            profile_level = level_values.get(profile.experience_level, 3)
            
            if profile_level >= required_level:
                level_score = 1.0
            else:
                # Penalty for being under-qualified
                level_score = max(0.2, profile_level / required_level)
        
        # Combined score (weighted equally)
        final_score = (years_score + level_score) / 2
        return MatchScore(final_score)

    def _calculate_education_score(
        self, 
        profile: Profile, 
        requirements: JobRequirements
    ) -> MatchScore:
        """Calculate education match score."""
        if not requirements.education_requirements:
            return MatchScore(1.0)
        
        # Simple education matching - can be enhanced
        if profile.profile_data.education:
            # Basic scoring: has education = good score
            return MatchScore(0.8)
        else:
            # No education but requirements exist
            return MatchScore(0.3)

    def _calculate_location_score(
        self, 
        profile: Profile, 
        requirements: JobRequirements
    ) -> MatchScore:
        """Calculate location match score."""
        if not requirements.location_requirements:
            return MatchScore(1.0)
        
        if not profile.profile_data.location:
            return MatchScore(0.5)  # Unknown location
        
        # Simple location matching - can be enhanced with geo-coding
        profile_location = (
            f"{profile.profile_data.location.city or ''} "
            f"{profile.profile_data.location.state or ''} "
            f"{profile.profile_data.location.country or ''}"
        ).lower().strip()
        
        if requirements.location_requirements.lower() in profile_location:
            return MatchScore(1.0)
        else:
            return MatchScore(0.3)  # Location mismatch

    def _find_matching_skills(
        self, 
        profile: Profile, 
        requirements: JobRequirements
    ) -> List[str]:
        """Find skills that match between profile and requirements."""
        profile_skills = set(skill.lower() for skill in profile.normalized_skills)
        all_required = set(skill.lower() for skill in 
                          requirements.required_skills + requirements.preferred_skills)
        
        matching = profile_skills.intersection(all_required)
        return list(matching)

    def _find_missing_skills(
        self, 
        profile: Profile, 
        requirements: JobRequirements
    ) -> List[str]:
        """Find required skills missing from profile."""
        profile_skills = set(skill.lower() for skill in profile.normalized_skills)
        required_skills = set(skill.lower() for skill in requirements.required_skills)
        
        missing = required_skills - profile_skills
        return list(missing)

    def _generate_match_reasons(
        self,
        profile: Profile,
        requirements: JobRequirements,
        skills_score: MatchScore,
        experience_score: MatchScore
    ) -> List[str]:
        """Generate human-readable reasons for the match score."""
        reasons = []
        
        if skills_score.value >= 0.8:
            reasons.append("Strong skills match")
        elif skills_score.value >= 0.6:
            reasons.append("Good skills match")
        else:
            reasons.append("Limited skills match")
        
        if experience_score.value >= 0.8:
            reasons.append("Meets experience requirements")
        elif experience_score.value >= 0.6:
            reasons.append("Partially meets experience requirements")
        else:
            reasons.append("Below experience requirements")
        
        total_experience = profile.profile_data.total_experience_years()
        if total_experience > 10:
            reasons.append("Highly experienced candidate")
        elif total_experience > 5:
            reasons.append("Experienced candidate")
        
        if profile.profile_data.education:
            reasons.append("Has relevant education")
        
        return reasons


__all__ = ["IMatchingService", "MatchingService", "JobRequirements", "MatchResult"]