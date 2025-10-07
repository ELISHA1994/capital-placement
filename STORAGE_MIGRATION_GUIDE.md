# File Storage Migration Guide: Local to S3

This guide explains how to migrate from local file storage to AWS S3 storage, leveraging the hexagonal architecture pattern implemented in this application.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Migration Steps](#migration-steps)
- [Configuration](#configuration)
- [Implementation](#implementation)
- [Testing](#testing)
- [Rollback](#rollback)
- [Production Deployment](#production-deployment)

## Overview

The application uses the **hexagonal architecture** pattern with a clean separation between domain interfaces (`IFileStorage`) and infrastructure implementations. This allows seamless swapping between storage backends without changing application code.

### Current Architecture

```
Domain Layer (Port)
  └─ IFileStorage interface
       │
       ├─ LocalFileStorageAdapter (current)
       └─ S3FileStorageAdapter (to be implemented)
```

### What Remains Unchanged

Thanks to hexagonal architecture, **NO CHANGES** are required to:
- ✅ `UploadApplicationService`
- ✅ API endpoints (`/upload`, `/reprocess`, etc.)
- ✅ Business logic
- ✅ Database models
- ✅ `IFileStorage` interface

## Prerequisites

1. **AWS Account** with S3 access
2. **S3 Bucket** created (e.g., `my-app-cv-uploads`)
3. **IAM Credentials** with S3 permissions:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "s3:PutObject",
           "s3:GetObject",
           "s3:DeleteObject",
           "s3:ListBucket",
           "s3:HeadBucket"
         ],
         "Resource": [
           "arn:aws:s3:::my-app-cv-uploads",
           "arn:aws:s3:::my-app-cv-uploads/*"
         ]
       }
     ]
   }
   ```

## Migration Steps

### Step 1: Install AWS SDK

Add boto3 to your project dependencies:

```bash
pip install boto3
```

Or add to `pyproject.toml`:

```toml
[project.dependencies]
boto3 = "^1.34.0"
```

### Step 2: Add S3 Configuration

**Update `.env`:**

```bash
# File Storage Configuration
FILE_STORAGE_TYPE=s3  # Change from 'local' to 's3'
FILE_STORAGE_PATH=uploads  # S3 prefix/folder path (not a filesystem path)

# S3-specific settings
AWS_S3_BUCKET=my-app-cv-uploads
AWS_REGION=us-east-1

# Option 1: Use access keys (not recommended for production)
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

# Option 2: Use IAM role (recommended for production)
# AWS_PROFILE=default
# Leave access key fields empty to use IAM role
```

**Update `app/core/config.py`:**

Add these fields to the `Settings` class:

```python
# File Storage
FILE_STORAGE_TYPE: str = Field(default="local", description="Storage type: local or s3")
FILE_STORAGE_PATH: str = Field(default="./storage/uploads", description="Local storage base path or S3 prefix")

# S3 Configuration
AWS_S3_BUCKET: Optional[str] = Field(default=None, description="S3 bucket name")
AWS_REGION: str = Field(default="us-east-1", description="AWS region")
AWS_ACCESS_KEY_ID: Optional[str] = Field(default=None, description="AWS access key ID")
AWS_SECRET_ACCESS_KEY: Optional[str] = Field(default=None, description="AWS secret access key")
AWS_PROFILE: Optional[str] = Field(default=None, description="AWS profile name for IAM role")
```

### Step 3: Create S3FileStorageAdapter

**Create file: `app/services/adapters/s3_file_storage_adapter.py`**

```python
"""S3 File Storage Adapter implementing IFileStorage interface."""

import structlog
from typing import Dict, Any
import boto3
from botocore.exceptions import ClientError

from app.domain.interfaces import IFileStorage
from app.core.config import get_settings

logger = structlog.get_logger(__name__)


class S3FileStorageAdapter(IFileStorage):
    """
    S3-based file storage with multi-tenant isolation.

    Implements the IFileStorage interface using AWS S3 as the backend.
    Files are stored with the pattern: {prefix}/{tenant_id}/{upload_id}/{filename}
    """

    def __init__(
        self,
        bucket_name: str,
        region: str,
        prefix: str = "uploads",
        access_key: str = None,
        secret_key: str = None,
    ):
        """
        Initialize S3 storage adapter.

        Args:
            bucket_name: S3 bucket name
            region: AWS region (e.g., 'us-east-1')
            prefix: S3 key prefix for all files (e.g., 'uploads')
            access_key: AWS access key ID (optional, uses IAM role if not provided)
            secret_key: AWS secret access key (optional)
        """
        self.bucket_name = bucket_name
        self.prefix = prefix

        # Initialize S3 client
        session_config = {"region_name": region}
        if access_key and secret_key:
            session_config["aws_access_key_id"] = access_key
            session_config["aws_secret_access_key"] = secret_key

        self.s3_client = boto3.client("s3", **session_config)

        logger.info(
            "S3 storage adapter initialized",
            bucket=bucket_name,
            region=region,
            prefix=prefix
        )

    def _build_s3_key(self, tenant_id: str, upload_id: str, filename: str = None) -> str:
        """
        Build S3 object key with tenant isolation.

        Pattern: {prefix}/{tenant_id}/{upload_id}/{filename}
        Example: uploads/tenant-123/upload-456/resume.pdf
        """
        if filename:
            return f"{self.prefix}/{tenant_id}/{upload_id}/{filename}"
        return f"{self.prefix}/{tenant_id}/{upload_id}/"

    async def save_file(
        self,
        tenant_id: str,
        upload_id: str,
        filename: str,
        content: bytes
    ) -> str:
        """
        Save file to S3.

        Args:
            tenant_id: Tenant identifier for isolation
            upload_id: Upload identifier
            filename: Original filename
            content: File content as bytes

        Returns:
            S3 URI (e.g., s3://bucket-name/uploads/tenant-123/upload-456/file.pdf)

        Raises:
            ClientError: If S3 operation fails
        """
        try:
            s3_key = self._build_s3_key(tenant_id, upload_id, filename)

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=content,
                Metadata={
                    "tenant_id": tenant_id,
                    "upload_id": upload_id,
                    "original_filename": filename,
                }
            )

            logger.info(
                "File saved to S3",
                bucket=self.bucket_name,
                key=s3_key,
                file_size=len(content)
            )

            return f"s3://{self.bucket_name}/{s3_key}"

        except ClientError as e:
            logger.error("S3 save failed", error=str(e), key=s3_key)
            raise

    async def retrieve_file(self, tenant_id: str, upload_id: str) -> bytes:
        """
        Retrieve file from S3.

        Args:
            tenant_id: Tenant identifier
            upload_id: Upload identifier

        Returns:
            File content as bytes

        Raises:
            FileNotFoundError: If file does not exist
            ClientError: If S3 operation fails
        """
        try:
            # List objects to find the file (we don't store filename separately)
            prefix = self._build_s3_key(tenant_id, upload_id)
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=1
            )

            if "Contents" not in response or len(response["Contents"]) == 0:
                raise FileNotFoundError(
                    f"File not found in S3 for tenant_id={tenant_id}, upload_id={upload_id}"
                )

            s3_key = response["Contents"][0]["Key"]

            # Get the object
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            content = obj["Body"].read()

            logger.info(
                "File retrieved from S3",
                key=s3_key,
                size=len(content)
            )

            return content

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise FileNotFoundError(
                    f"File not found in S3: tenant_id={tenant_id}, upload_id={upload_id}"
                )
            logger.error("S3 retrieve failed", error=str(e))
            raise

    async def delete_file(self, tenant_id: str, upload_id: str) -> bool:
        """
        Delete file from S3.

        Deletes all objects matching the upload_id prefix (handles multiple files).

        Args:
            tenant_id: Tenant identifier
            upload_id: Upload identifier

        Returns:
            True if files were deleted, False if no files found

        Raises:
            ClientError: If S3 operation fails
        """
        try:
            prefix = self._build_s3_key(tenant_id, upload_id)

            # List all objects with this prefix
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )

            if "Contents" not in response:
                logger.info("No files found to delete", prefix=prefix)
                return False

            # Delete all objects
            deleted_count = 0
            for obj in response["Contents"]:
                self.s3_client.delete_object(
                    Bucket=self.bucket_name,
                    Key=obj["Key"]
                )
                deleted_count += 1

            logger.info(
                "Files deleted from S3",
                prefix=prefix,
                deleted_count=deleted_count
            )
            return True

        except ClientError as e:
            logger.error("S3 delete failed", error=str(e))
            raise

    async def exists(self, tenant_id: str, upload_id: str) -> bool:
        """
        Check if file exists in S3.

        Args:
            tenant_id: Tenant identifier
            upload_id: Upload identifier

        Returns:
            True if file exists, False otherwise
        """
        try:
            prefix = self._build_s3_key(tenant_id, upload_id)
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=1
            )
            exists = "Contents" in response and len(response["Contents"]) > 0

            logger.debug("File existence check", prefix=prefix, exists=exists)
            return exists

        except ClientError as e:
            logger.error("S3 existence check failed", error=str(e))
            return False

    async def check_health(self) -> Dict[str, Any]:
        """
        Health check - verify S3 bucket access.

        Returns:
            Health status dict with service status and bucket info
        """
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            return {
                "service": "s3_file_storage",
                "status": "healthy",
                "bucket": self.bucket_name,
                "region": self.s3_client.meta.region_name,
            }
        except ClientError as e:
            return {
                "service": "s3_file_storage",
                "status": "unhealthy",
                "bucket": self.bucket_name,
                "error": str(e),
                "error_code": e.response["Error"]["Code"],
            }
```

### Step 4: Update Storage Provider

**Update `app/infrastructure/providers/storage_provider.py`:**

Replace the `get_file_storage()` function with:

```python
async def get_file_storage() -> IFileStorage:
    """
    Get file storage service (singleton).

    Returns the appropriate storage adapter based on FILE_STORAGE_TYPE configuration:
    - 'local': LocalFileStorageAdapter (filesystem)
    - 's3': S3FileStorageAdapter (AWS S3)

    Returns:
        IFileStorage: Configured storage adapter instance

    Raises:
        ValueError: If storage type is unsupported or required config is missing
    """
    global _file_storage

    if _file_storage is None:
        settings = get_settings()

        if settings.FILE_STORAGE_TYPE == "local":
            from app.services.adapters.local_file_storage_adapter import LocalFileStorageAdapter

            _file_storage = LocalFileStorageAdapter(
                base_path=settings.FILE_STORAGE_PATH
            )
            logger.info(
                "Initialized local file storage",
                path=settings.FILE_STORAGE_PATH
            )

        elif settings.FILE_STORAGE_TYPE == "s3":
            from app.services.adapters.s3_file_storage_adapter import S3FileStorageAdapter

            # Validate S3 configuration
            if not settings.AWS_S3_BUCKET:
                raise ValueError(
                    "AWS_S3_BUCKET must be set when FILE_STORAGE_TYPE=s3"
                )

            _file_storage = S3FileStorageAdapter(
                bucket_name=settings.AWS_S3_BUCKET,
                region=settings.AWS_REGION,
                prefix=settings.FILE_STORAGE_PATH,
                access_key=settings.AWS_ACCESS_KEY_ID,
                secret_key=settings.AWS_SECRET_ACCESS_KEY,
            )
            logger.info(
                "Initialized S3 file storage",
                bucket=settings.AWS_S3_BUCKET,
                region=settings.AWS_REGION,
                prefix=settings.FILE_STORAGE_PATH
            )

        else:
            raise ValueError(
                f"Unsupported storage type: {settings.FILE_STORAGE_TYPE}. "
                f"Supported types: 'local', 's3'"
            )

    return _file_storage
```

### Step 5: Update Exports (Optional)

**Update `app/services/adapters/__init__.py`** to export the new adapter:

```python
from .local_file_storage_adapter import LocalFileStorageAdapter
from .s3_file_storage_adapter import S3FileStorageAdapter

__all__ = [
    "LocalFileStorageAdapter",
    "S3FileStorageAdapter",
]
```

## Configuration

### Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `FILE_STORAGE_TYPE` | Yes | Storage backend type | `s3` |
| `FILE_STORAGE_PATH` | Yes | S3 prefix/folder | `uploads` |
| `AWS_S3_BUCKET` | Yes (for S3) | S3 bucket name | `my-app-uploads` |
| `AWS_REGION` | Yes | AWS region | `us-east-1` |
| `AWS_ACCESS_KEY_ID` | No* | AWS access key | `AKIAIOSFODNN7EXAMPLE` |
| `AWS_SECRET_ACCESS_KEY` | No* | AWS secret key | `wJalrXUt...` |
| `AWS_PROFILE` | No* | AWS profile name | `production` |

*Either provide access keys OR use IAM role/profile

### S3 Bucket Structure

Files will be organized as:

```
s3://my-app-uploads/
└── uploads/                          (FILE_STORAGE_PATH)
    ├── tenant-123/                   (tenant_id)
    │   ├── upload-456/               (upload_id)
    │   │   └── resume.pdf            (filename)
    │   └── upload-789/
    │       └── cv.docx
    └── tenant-999/
        └── upload-abc/
            └── document.pdf
```

## Testing

### 1. Test S3 Adapter Directly

Create a test script `test_s3_storage.py`:

```python
import asyncio
from app.infrastructure.providers.storage_provider import get_file_storage

async def test_s3():
    storage = await get_file_storage()

    # Test save
    test_content = b"Hello, S3!"
    path = await storage.save_file(
        tenant_id="test-tenant",
        upload_id="test-upload",
        filename="test.txt",
        content=test_content
    )
    print(f"Saved: {path}")

    # Test exists
    exists = await storage.exists("test-tenant", "test-upload")
    print(f"Exists: {exists}")

    # Test retrieve
    content = await storage.retrieve_file("test-tenant", "test-upload")
    print(f"Retrieved: {content.decode()}")

    # Test delete
    deleted = await storage.delete_file("test-tenant", "test-upload")
    print(f"Deleted: {deleted}")

    # Test health
    health = await storage.check_health()
    print(f"Health: {health}")

asyncio.run(test_s3())
```

### 2. Test Upload Endpoint

```bash
curl -X POST "http://localhost:8000/api/v1/upload/" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@resume.pdf" \
  -F "auto_process=true"
```

### 3. Test Reprocess Endpoint

```bash
curl -X POST "http://localhost:8000/api/v1/upload/{upload_id}/reprocess" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

Verify in CloudWatch or application logs that S3 operations are working.

## Rollback

If you need to rollback to local storage:

1. **Update `.env`:**
   ```bash
   FILE_STORAGE_TYPE=local
   FILE_STORAGE_PATH=./storage/uploads
   ```

2. **Restart the application:**
   ```bash
   pkill -f "uvicorn app.main:app"
   python -m uvicorn app.main:app --reload --port 8000
   ```

3. **No code changes needed!** The provider will automatically switch back to `LocalFileStorageAdapter`.

## Migrating Existing Files

If you have existing files in local storage and want to migrate them to S3:

### Option 1: AWS CLI Sync

```bash
# Sync local storage to S3
aws s3 sync ./storage/uploads s3://my-app-uploads/uploads/ \
  --acl private \
  --storage-class STANDARD \
  --exclude "*.DS_Store"

# Verify sync
aws s3 ls s3://my-app-uploads/uploads/ --recursive --summarize
```

### Option 2: Migration Script

Create `migrate_to_s3.py`:

```python
import asyncio
import os
from pathlib import Path
from app.infrastructure.providers.storage_provider import get_file_storage

async def migrate():
    storage = await get_file_storage()
    base_path = Path("./storage/uploads")

    migrated = 0
    errors = 0

    for tenant_dir in base_path.iterdir():
        if not tenant_dir.is_dir():
            continue

        tenant_id = tenant_dir.name

        for upload_dir in tenant_dir.iterdir():
            if not upload_dir.is_dir():
                continue

            upload_id = upload_dir.name

            for file_path in upload_dir.iterdir():
                if file_path.is_file():
                    try:
                        with open(file_path, 'rb') as f:
                            content = f.read()

                        await storage.save_file(
                            tenant_id=tenant_id,
                            upload_id=upload_id,
                            filename=file_path.name,
                            content=content
                        )

                        print(f"✓ Migrated: {tenant_id}/{upload_id}/{file_path.name}")
                        migrated += 1
                    except Exception as e:
                        print(f"✗ Error: {file_path}: {e}")
                        errors += 1

    print(f"\nMigration complete: {migrated} files migrated, {errors} errors")

asyncio.run(migrate())
```

## Production Deployment

### Recommended Setup

1. **Use IAM Roles** instead of access keys (EC2, ECS, Lambda)
2. **Enable S3 versioning** for data protection
3. **Set lifecycle policies** for cost optimization
4. **Enable S3 access logging** for audit trails
5. **Use VPC endpoints** for S3 to avoid internet traffic

### Environment Configuration

**Development:**
```bash
FILE_STORAGE_TYPE=local
FILE_STORAGE_PATH=./storage/uploads
```

**Staging:**
```bash
FILE_STORAGE_TYPE=s3
FILE_STORAGE_PATH=uploads
AWS_S3_BUCKET=myapp-staging-uploads
AWS_REGION=us-east-1
# Use IAM role, no keys needed
```

**Production:**
```bash
FILE_STORAGE_TYPE=s3
FILE_STORAGE_PATH=uploads
AWS_S3_BUCKET=myapp-prod-uploads
AWS_REGION=us-east-1
# Use IAM role, no keys needed
```

### S3 Bucket Policies

**Enable encryption:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyUnencryptedObjectUploads",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:PutObject",
      "Resource": "arn:aws:s3:::my-app-uploads/*",
      "Condition": {
        "StringNotEquals": {
          "s3:x-amz-server-side-encryption": "AES256"
        }
      }
    }
  ]
}
```

**Enable versioning and lifecycle:**
```bash
aws s3api put-bucket-versioning \
  --bucket my-app-uploads \
  --versioning-configuration Status=Enabled

aws s3api put-bucket-lifecycle-configuration \
  --bucket my-app-uploads \
  --lifecycle-configuration file://lifecycle.json
```

## Troubleshooting

### Issue: "NoSuchBucket" error

**Solution:** Verify bucket exists and region is correct
```bash
aws s3 ls s3://my-app-uploads
```

### Issue: "AccessDenied" error

**Solution:** Check IAM permissions
```bash
aws s3 cp test.txt s3://my-app-uploads/test.txt
aws s3 rm s3://my-app-uploads/test.txt
```

### Issue: Files not found after migration

**Solution:** Verify S3 key structure matches expected pattern
```bash
aws s3 ls s3://my-app-uploads/uploads/ --recursive
```

### Issue: High S3 costs

**Solution:**
- Enable S3 Intelligent-Tiering
- Set lifecycle policies to move old files to Glacier
- Monitor usage with AWS Cost Explorer

## Summary

### What You Need to Do

1. ✅ Install boto3
2. ✅ Add S3 config to `.env` and `config.py`
3. ✅ Create `S3FileStorageAdapter` class
4. ✅ Update `storage_provider.py`
5. ✅ Change `FILE_STORAGE_TYPE=s3` in `.env`
6. ✅ Restart application

### What You DON'T Need to Change

- ❌ Upload service code
- ❌ API endpoints
- ❌ Business logic
- ❌ Database models
- ❌ Tests (they use the interface)

**Total effort: ~30 minutes** ⚡

The power of hexagonal architecture! 🎯