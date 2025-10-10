"""Export-related DTOs for API layer.

Provides request/response schemas for profile export operations
following Pydantic best practices and API-layer conventions.
"""


from pydantic import BaseModel, Field

from app.domain.entities.profile import ProcessingStatus


class ProfileExportRequest(BaseModel):
    """CSV export configuration request.

    Defines filters and field selection for profile CSV export.
    Used as query parameters in the export endpoint.
    """

    status_filter: ProcessingStatus | None = Field(
        None,
        description="Filter profiles by processing status (pending, processing, completed, failed, partial, cancelled)",
    )
    skill_filter: str | None = Field(
        None,
        description="Filter profiles by skill (partial match, case-insensitive)",
        example="python",
    )
    include_contact_info: bool = Field(
        False,
        description="Include contact information (email, phone, location) - privacy setting",
    )
    include_full_text: bool = Field(
        False,
        description="Include full text fields (summary, searchable text) - may increase file size",
    )


class ProfileExportMetadata(BaseModel):
    """Metadata about CSV export operation.

    Provides information about the export result for logging
    and user feedback purposes.
    """

    total_profiles: int = Field(
        ...,
        description="Total number of profiles exported",
        ge=0,
    )
    filters_applied: dict[str, str] = Field(
        default_factory=dict,
        description="Filters that were applied during export",
    )
    fields_included: list[str] = Field(
        default_factory=list,
        description="List of fields included in the export",
    )
    generated_at: str = Field(
        ...,
        description="ISO timestamp when export was generated",
    )


__all__ = [
    "ProfileExportRequest",
    "ProfileExportMetadata",
]
