"""Domain service for profile quality scoring and analysis."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any

from app.domain.entities.profile import Profile, Skill, Experience, Education


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for a profile."""
    
    overall_score: float  # 0-100
    completeness_score: float  # 0-100
    consistency_score: float  # 0-100
    richness_score: float  # 0-100
    recency_score: float  # 0-100
    issues: List[str]
    recommendations: List[str]
    strengths: List[str]

    @property
    def is_high_quality(self) -> bool:
        """Determine if profile is high quality."""
        return self.overall_score >= 80

    @property
    def is_searchable_quality(self) -> bool:
        """Determine if profile meets minimum quality for search."""
        return self.overall_score >= 60


class IProfileScoringService(ABC):
    """Domain service interface for profile quality assessment."""

    @abstractmethod
    def calculate_quality_metrics(self, profile: Profile) -> QualityMetrics:
        """Calculate comprehensive quality metrics for a profile."""
        pass

    @abstractmethod
    def identify_quality_issues(self, profile: Profile) -> List[str]:
        """Identify specific quality issues in a profile."""
        pass

    @abstractmethod
    def generate_improvement_recommendations(self, profile: Profile) -> List[str]:
        """Generate recommendations to improve profile quality."""
        pass


class ProfileScoringService(IProfileScoringService):
    """Concrete implementation of profile quality scoring."""

    def calculate_quality_metrics(self, profile: Profile) -> QualityMetrics:
        """Calculate comprehensive quality assessment."""
        completeness = self._calculate_completeness_score(profile)
        consistency = self._calculate_consistency_score(profile)
        richness = self._calculate_richness_score(profile)
        recency = self._calculate_recency_score(profile)
        
        # Weighted overall score
        overall = (
            completeness * 0.4 +
            consistency * 0.2 +
            richness * 0.3 +
            recency * 0.1
        )
        
        issues = self.identify_quality_issues(profile)
        recommendations = self.generate_improvement_recommendations(profile)
        strengths = self._identify_strengths(profile)
        
        return QualityMetrics(
            overall_score=overall,
            completeness_score=completeness,
            consistency_score=consistency,
            richness_score=richness,
            recency_score=recency,
            issues=issues,
            recommendations=recommendations,
            strengths=strengths
        )

    def identify_quality_issues(self, profile: Profile) -> List[str]:
        """Identify specific quality issues."""
        issues = []
        
        # Core field issues
        if not profile.profile_data.name:
            issues.append("Missing name")
        if not profile.profile_data.email:
            issues.append("Missing email")
        if not profile.profile_data.summary:
            issues.append("Missing professional summary")
        
        # Experience issues
        if not profile.profile_data.experience:
            issues.append("No work experience listed")
        else:
            for i, exp in enumerate(profile.profile_data.experience):
                if not exp.description or len(exp.description) < 50:
                    issues.append(f"Experience #{i+1} has insufficient description")
                if exp.current and exp.end_date:
                    issues.append(f"Experience #{i+1} marked as current but has end date")
        
        # Skills issues
        if not profile.profile_data.skills:
            issues.append("No skills listed")
        elif len(profile.profile_data.skills) < 3:
            issues.append("Very few skills listed (less than 3)")
        
        # Education issues
        if not profile.profile_data.education:
            issues.append("No education listed")
        
        # Contact issues
        if not profile.profile_data.phone:
            issues.append("No phone number provided")
        if not profile.profile_data.location:
            issues.append("No location information")
        elif not profile.profile_data.location.is_complete():
            issues.append("Incomplete location information")
        
        # Processing issues
        if not profile.embeddings.has_complete_embeddings():
            issues.append("Missing AI embeddings")
        
        # Privacy issues
        if not profile.privacy.consent_given:
            issues.append("No data processing consent")
        
        return issues

    def generate_improvement_recommendations(self, profile: Profile) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Completeness recommendations
        if not profile.profile_data.summary:
            recommendations.append("Add a professional summary highlighting key achievements")
        elif len(profile.profile_data.summary) < 100:
            recommendations.append("Expand professional summary (currently too brief)")
        
        if len(profile.profile_data.skills) < 10:
            recommendations.append("Add more skills to improve searchability")
        
        if not profile.profile_data.education:
            recommendations.append("Add education information if applicable")
        
        # Content quality recommendations
        if profile.profile_data.experience:
            short_descriptions = [
                exp for exp in profile.profile_data.experience 
                if not exp.description or len(exp.description) < 100
            ]
            if short_descriptions:
                recommendations.append("Expand work experience descriptions with specific achievements")
        
        # Skills recommendations
        unskilled_entries = [
            skill for skill in profile.profile_data.skills 
            if not skill.proficiency and not skill.years_of_experience
        ]
        if unskilled_entries:
            recommendations.append("Add proficiency levels or experience years for skills")
        
        # Contact recommendations
        if not profile.profile_data.phone:
            recommendations.append("Add phone number for better recruiter contact")
        
        if not profile.profile_data.location or not profile.profile_data.location.is_complete():
            recommendations.append("Complete location information for location-based searches")
        
        # Professional recommendations
        if profile.profile_data.total_experience_years() > 2 and not any(
            exp.achievements for exp in profile.profile_data.experience
        ):
            recommendations.append("Add specific achievements and accomplishments to work experience")
        
        return recommendations

    def _calculate_completeness_score(self, profile: Profile) -> float:
        """Calculate how complete the profile is."""
        score = 0.0
        max_score = 100.0
        
        # Essential fields (50 points)
        if profile.profile_data.name: score += 10
        if profile.profile_data.email: score += 10
        if profile.profile_data.summary: score += 15
        if profile.profile_data.experience: score += 15
        
        # Important fields (30 points)
        if profile.profile_data.skills: score += 10
        if profile.profile_data.education: score += 10
        if profile.profile_data.location and profile.profile_data.location.is_complete(): score += 10
        
        # Nice-to-have fields (20 points)
        if profile.profile_data.phone: score += 5
        if profile.profile_data.headline: score += 5
        if profile.profile_data.languages: score += 5
        if profile.embeddings.has_complete_embeddings(): score += 5
        
        return min(score, max_score)

    def _calculate_consistency_score(self, profile: Profile) -> float:
        """Calculate internal consistency of profile data."""
        score = 100.0  # Start with perfect score, deduct for issues
        
        # Check for date inconsistencies
        for exp in profile.profile_data.experience:
            if exp.current and exp.end_date:
                score -= 10  # Current job shouldn't have end date
        
        # Check for skill-experience consistency
        exp_skills = set()
        for exp in profile.profile_data.experience:
            exp_skills.update(skill.name.normalized for skill in exp.skills)
        
        profile_skills = set(profile.normalized_skills)
        if exp_skills and not exp_skills.intersection(profile_skills):
            score -= 15  # No overlap between experience skills and profile skills
        
        # Check for reasonable experience levels
        total_years = profile.profile_data.total_experience_years()
        if total_years > 50:  # Unreasonably long career
            score -= 20
        
        return max(0.0, score)

    def _calculate_richness_score(self, profile: Profile) -> float:
        """Calculate how rich and detailed the profile content is."""
        score = 0.0
        
        # Summary richness (25 points)
        if profile.profile_data.summary:
            summary_len = len(profile.profile_data.summary)
            if summary_len >= 200: score += 25
            elif summary_len >= 100: score += 15
            elif summary_len >= 50: score += 10
        
        # Experience richness (35 points)
        if profile.profile_data.experience:
            exp_score = 0
            for exp in profile.profile_data.experience:
                if exp.description and len(exp.description) >= 100: exp_score += 5
                if exp.achievements: exp_score += 3
                if exp.skills: exp_score += 2
            score += min(35, exp_score)
        
        # Skills richness (20 points)
        skill_count = len(profile.profile_data.skills)
        if skill_count >= 15: score += 20
        elif skill_count >= 10: score += 15
        elif skill_count >= 5: score += 10
        elif skill_count >= 3: score += 5
        
        # Additional content (20 points)
        if profile.profile_data.education: score += 10
        if profile.profile_data.languages: score += 5
        if len(profile.profile_data.experience) >= 3: score += 5
        
        return min(100.0, score)

    def _calculate_recency_score(self, profile: Profile) -> float:
        """Calculate how recent and up-to-date the profile is."""
        score = 100.0
        
        # Check if there's current employment
        has_current_job = any(exp.is_current_role() for exp in profile.profile_data.experience)
        if not has_current_job:
            score -= 20  # No current employment
        
        # Check profile update recency
        days_since_update = (profile.updated_at - profile.created_at).days
        if days_since_update > 365:
            score -= 30  # Very old profile
        elif days_since_update > 180:
            score -= 15  # Somewhat old profile
        
        # Check processing status
        if profile.processing.status.value != "completed":
            score -= 10  # Not fully processed
        
        return max(0.0, score)

    def _identify_strengths(self, profile: Profile) -> List[str]:
        """Identify profile strengths."""
        strengths = []
        
        # Content strengths
        if profile.profile_data.summary and len(profile.profile_data.summary) >= 200:
            strengths.append("Comprehensive professional summary")
        
        if len(profile.profile_data.skills) >= 15:
            strengths.append("Extensive skills listing")
        
        if len(profile.profile_data.experience) >= 3:
            strengths.append("Rich work experience history")
        
        # Achievement strengths
        achievement_count = sum(
            len(exp.achievements) for exp in profile.profile_data.experience
        )
        if achievement_count >= 5:
            strengths.append("Well-documented achievements")
        
        # Technical strengths
        if profile.embeddings.has_complete_embeddings():
            strengths.append("AI-ready with vector embeddings")
        
        # Experience strengths
        total_years = profile.profile_data.total_experience_years()
        if total_years >= 10:
            strengths.append("Highly experienced professional")
        elif total_years >= 5:
            strengths.append("Experienced professional")
        
        # Education strengths
        if profile.profile_data.education:
            degree_count = len(profile.profile_data.education)
            if degree_count >= 2:
                strengths.append("Multiple educational qualifications")
            else:
                strengths.append("Educational background")
        
        return strengths


__all__ = ["IProfileScoringService", "ProfileScoringService", "QualityMetrics"]