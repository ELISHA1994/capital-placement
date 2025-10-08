"""
Shared API response types and base schemas.

This module contains common API response models that are used across multiple
endpoints. These are DTOs (Data Transfer Objects) in the API layer, separate
from domain entities and persistence tables.

Following hexagonal architecture principles:
- API Layer: HTTP request/response DTOs (this file)
- Domain Layer: Business entities and logic
- Infrastructure Layer: Database tables and adapters
"""

from typing import Any, List

from pydantic import Field
from sqlmodel import SQLModel


class PaginatedResponse(SQLModel):
    """
    Generic paginated response model for API endpoints.

    This DTO provides a consistent pagination structure across all API endpoints
    that return paginated data. It includes metadata about the current page,
    total items, and navigation helpers.

    Attributes:
        items: List of items for the current page
        total: Total number of items across all pages
        page: Current page number (1-based)
        size: Number of items per page
        pages: Total number of pages
        has_next: Whether there is a next page available
        has_prev: Whether there is a previous page available

    Example:
        ```python
        response = PaginatedResponse.create(
            items=[profile1, profile2, profile3],
            total=100,
            page=1,
            size=20
        )
        # response.pages = 5
        # response.has_next = True
        # response.has_prev = False
        ```
    """

    items: List[Any] = Field(
        default_factory=list,
        description="List of items for current page"
    )
    total: int = Field(
        default=0,
        ge=0,
        description="Total number of items across all pages"
    )
    page: int = Field(
        default=1,
        ge=1,
        description="Current page number"
    )
    size: int = Field(
        default=20,
        ge=1,
        description="Items per page"
    )
    pages: int = Field(
        default=0,
        ge=0,
        description="Total number of pages"
    )
    has_next: bool = Field(
        default=False,
        description="Whether there is a next page"
    )
    has_prev: bool = Field(
        default=False,
        description="Whether there is a previous page"
    )

    @classmethod
    def create(
        cls,
        items: List[Any],
        total: int,
        page: int,
        size: int
    ) -> "PaginatedResponse":
        """
        Create a paginated response from query results.

        This factory method automatically calculates pagination metadata
        including total pages and navigation flags.

        Args:
            items: List of items for the current page
            total: Total number of items across all pages
            page: Current page number (1-based)
            size: Number of items per page

        Returns:
            PaginatedResponse with all fields populated

        Example:
            ```python
            # After fetching items from database
            response = PaginatedResponse.create(
                items=fetched_items,
                total=count_query_result,
                page=request.page,
                size=request.size
            )
            ```
        """
        pages = (total + size - 1) // size  # Ceiling division

        return cls(
            items=items,
            total=total,
            page=page,
            size=size,
            pages=pages,
            has_next=page < pages,
            has_prev=page > 1
        )