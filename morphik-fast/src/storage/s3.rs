use async_trait::async_trait;
use aws_sdk_s3::Client;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing::{error, info};

use super::{Storage, StorageError};

/// AWS S3 storage backend.
pub struct S3Storage {
    client: Client,
    pub default_bucket_name: String,
    upload_semaphore: Arc<Semaphore>,
    region: String,
}

impl S3Storage {
    pub async fn new(
        aws_access_key: &str,
        aws_secret_key: &str,
        region: &str,
        default_bucket: &str,
        upload_concurrency: u32,
    ) -> Result<Self, StorageError> {
        let creds = aws_sdk_s3::config::Credentials::new(
            aws_access_key,
            aws_secret_key,
            None,
            None,
            "morphik-fast",
        );

        let config = aws_sdk_s3::config::Builder::new()
            .region(aws_sdk_s3::config::Region::new(region.to_string()))
            .credentials_provider(creds)
            .build();

        let client = Client::from_conf(config);

        Ok(Self {
            client,
            default_bucket_name: default_bucket.to_string(),
            upload_semaphore: Arc::new(Semaphore::new(upload_concurrency.max(1) as usize)),
            region: region.to_string(),
        })
    }

    async fn ensure_bucket(&self, bucket: &str) -> Result<(), StorageError> {
        match self.client.head_bucket().bucket(bucket).send().await {
            Ok(_) => Ok(()),
            Err(_) => {
                let mut req = self.client.create_bucket().bucket(bucket);
                if self.region != "us-east-1" {
                    let constraint = aws_sdk_s3::types::CreateBucketConfiguration::builder()
                        .location_constraint(
                            aws_sdk_s3::types::BucketLocationConstraint::from(
                                self.region.as_str(),
                            ),
                        )
                        .build();
                    req = req.create_bucket_configuration(constraint);
                }
                match req.send().await {
                    Ok(_) => Ok(()),
                    Err(e) => {
                        let msg = format!("{e}");
                        if msg.contains("BucketAlreadyOwnedByYou")
                            || msg.contains("BucketAlreadyExists")
                        {
                            Ok(())
                        } else {
                            Err(StorageError::S3(msg))
                        }
                    }
                }
            }
        }
    }

    fn resolve_bucket<'a>(&'a self, bucket: &'a str) -> &'a str {
        if bucket.is_empty() {
            &self.default_bucket_name
        } else {
            bucket
        }
    }
}

#[async_trait]
impl Storage for S3Storage {
    async fn upload_from_base64(
        &self,
        content: &str,
        key: &str,
        content_type: Option<&str>,
        bucket: &str,
    ) -> Result<(String, String), StorageError> {
        use base64::Engine;

        // Handle data URI
        let payload = if content.starts_with("data:") {
            content
                .split_once(',')
                .map(|(_, b)| b)
                .unwrap_or(content)
        } else {
            content
        };

        let decoded = base64::engine::general_purpose::STANDARD.decode(payload)?;
        self.upload_bytes(&decoded, key, content_type, bucket).await
    }

    async fn upload_bytes(
        &self,
        data: &[u8],
        key: &str,
        content_type: Option<&str>,
        bucket: &str,
    ) -> Result<(String, String), StorageError> {
        let target_bucket = self.resolve_bucket(bucket);
        let _permit = self.upload_semaphore.acquire().await.map_err(|e| {
            StorageError::Other(format!("Semaphore error: {e}"))
        })?;

        self.ensure_bucket(target_bucket).await?;

        let body = aws_sdk_s3::primitives::ByteStream::from(data.to_vec());
        let mut req = self
            .client
            .put_object()
            .bucket(target_bucket)
            .key(key)
            .body(body);

        if let Some(ct) = content_type {
            req = req.content_type(ct);
        }

        req.send().await.map_err(|e| {
            error!("S3 upload error: {e}");
            StorageError::S3(format!("{e}"))
        })?;

        Ok((target_bucket.to_string(), key.to_string()))
    }

    async fn download_file(&self, bucket: &str, key: &str) -> Result<Vec<u8>, StorageError> {
        let target_bucket = self.resolve_bucket(bucket);
        let resp = self
            .client
            .get_object()
            .bucket(target_bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| {
                let msg = format!("{e}");
                if msg.contains("NoSuchKey") || msg.contains("404") {
                    StorageError::NotFound {
                        bucket: target_bucket.to_string(),
                        key: key.to_string(),
                    }
                } else {
                    StorageError::S3(msg)
                }
            })?;

        let bytes = resp.body.collect().await.map_err(|e| {
            StorageError::S3(format!("Failed to read S3 body: {e}"))
        })?;

        Ok(bytes.to_vec())
    }

    async fn get_download_url(
        &self,
        bucket: &str,
        key: &str,
        expires_in: u64,
    ) -> Result<String, StorageError> {
        if bucket.is_empty() || key.is_empty() {
            return Ok(String::new());
        }

        let target_bucket = self.resolve_bucket(bucket);
        let presigned = self
            .client
            .get_object()
            .bucket(target_bucket)
            .key(key)
            .presigned(
                aws_sdk_s3::presigning::PresigningConfig::expires_in(
                    std::time::Duration::from_secs(expires_in),
                )
                .map_err(|e| StorageError::S3(format!("Presign config error: {e}")))?,
            )
            .await
            .map_err(|e| StorageError::S3(format!("Presign error: {e}")))?;

        Ok(presigned.uri().to_string())
    }

    async fn delete_file(&self, bucket: &str, key: &str) -> Result<bool, StorageError> {
        let target_bucket = self.resolve_bucket(bucket);
        self.client
            .delete_object()
            .bucket(target_bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| {
                error!("S3 delete error: {e}");
                StorageError::S3(format!("{e}"))
            })?;
        info!("Deleted {key} from bucket {target_bucket}");
        Ok(true)
    }

    async fn get_object_size(&self, bucket: &str, key: &str) -> Result<u64, StorageError> {
        let target_bucket = self.resolve_bucket(bucket);
        let resp = self
            .client
            .head_object()
            .bucket(target_bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| StorageError::S3(format!("Head object error: {e}")))?;

        Ok(resp.content_length().unwrap_or(0) as u64)
    }

    fn provider_name(&self) -> &str {
        "aws-s3"
    }

    fn default_bucket(&self) -> &str {
        &self.default_bucket_name
    }
}
