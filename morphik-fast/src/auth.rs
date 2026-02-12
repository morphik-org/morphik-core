use axum::http::StatusCode;
use jsonwebtoken::{decode, DecodingKey, Validation};
use serde::{Deserialize, Serialize};

use crate::models::api::AuthContext;

/// JWT claims structure.
#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    /// User ID.
    pub user_id: Option<String>,
    /// Backward-compat: entity_id.
    pub entity_id: Option<String>,
    /// Application ID.
    pub app_id: Option<String>,
    /// Token version for revocation.
    #[serde(default)]
    pub token_version: u32,
    /// Expiration time (Unix timestamp).
    pub exp: Option<u64>,
}

/// Verify a JWT token and extract auth context.
pub fn verify_token(
    token: &str,
    secret: &str,
    algorithm: &str,
) -> Result<AuthContext, String> {
    let algo = match algorithm {
        "HS256" => jsonwebtoken::Algorithm::HS256,
        "HS384" => jsonwebtoken::Algorithm::HS384,
        "HS512" => jsonwebtoken::Algorithm::HS512,
        _ => return Err(format!("Unsupported algorithm: {algorithm}")),
    };

    let mut validation = Validation::new(algo);
    // Allow some clock drift.
    validation.leeway = 60;
    // Don't require specific claims.
    validation.required_spec_claims = std::collections::HashSet::new();

    let key = DecodingKey::from_secret(secret.as_bytes());
    let token_data = decode::<Claims>(token, &key, &validation)
        .map_err(|e| format!("Token validation failed: {e}"))?;

    let claims = token_data.claims;
    let user_id = claims
        .user_id
        .or(claims.entity_id)
        .unwrap_or_else(|| "unknown".to_string());

    Ok(AuthContext {
        user_id,
        app_id: claims.app_id,
        permissions: vec![],
    })
}

/// Extract auth context from an Authorization header.
pub fn extract_auth_from_header(
    auth_header: Option<&str>,
    secret: &str,
    algorithm: &str,
    bypass_mode: bool,
    dev_user_id: &str,
) -> Result<AuthContext, (StatusCode, String)> {
    if bypass_mode {
        return Ok(AuthContext {
            user_id: dev_user_id.to_string(),
            app_id: Some(dev_user_id.to_string()),
            permissions: vec![],
        });
    }

    let header = auth_header.ok_or_else(|| {
        (
            StatusCode::UNAUTHORIZED,
            "Missing Authorization header".to_string(),
        )
    })?;

    let token = header
        .strip_prefix("Bearer ")
        .ok_or_else(|| {
            (
                StatusCode::UNAUTHORIZED,
                "Invalid Authorization header format".to_string(),
            )
        })?;

    verify_token(token, secret, algorithm).map_err(|e| (StatusCode::UNAUTHORIZED, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use jsonwebtoken::{encode, EncodingKey, Header};

    fn make_token(claims: &Claims, secret: &str) -> String {
        encode(
            &Header::default(),
            claims,
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap()
    }

    #[test]
    fn test_verify_valid_token() {
        let claims = Claims {
            user_id: Some("user1".to_string()),
            entity_id: None,
            app_id: Some("app1".to_string()),
            token_version: 0,
            exp: Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
                    + 3600,
            ),
        };
        let token = make_token(&claims, "secret");
        let auth = verify_token(&token, "secret", "HS256").unwrap();
        assert_eq!(auth.user_id, "user1");
        assert_eq!(auth.app_id, Some("app1".to_string()));
    }

    #[test]
    fn test_verify_invalid_secret() {
        let claims = Claims {
            user_id: Some("user1".to_string()),
            entity_id: None,
            app_id: None,
            token_version: 0,
            exp: Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
                    + 3600,
            ),
        };
        let token = make_token(&claims, "secret");
        let result = verify_token(&token, "wrong-secret", "HS256");
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_entity_id_fallback() {
        let claims = Claims {
            user_id: None,
            entity_id: Some("entity1".to_string()),
            app_id: None,
            token_version: 0,
            exp: Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
                    + 3600,
            ),
        };
        let token = make_token(&claims, "secret");
        let auth = verify_token(&token, "secret", "HS256").unwrap();
        assert_eq!(auth.user_id, "entity1");
    }

    #[test]
    fn test_bypass_auth_mode() {
        let result = extract_auth_from_header(None, "secret", "HS256", true, "dev_user");
        assert!(result.is_ok());
        let auth = result.unwrap();
        assert_eq!(auth.user_id, "dev_user");
        assert_eq!(auth.app_id, Some("dev_user".to_string()));
    }

    #[test]
    fn test_missing_header_no_bypass() {
        let result = extract_auth_from_header(None, "secret", "HS256", false, "dev_user");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().0, StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn test_invalid_header_format() {
        let result =
            extract_auth_from_header(Some("Basic abc"), "secret", "HS256", false, "dev_user");
        assert!(result.is_err());
    }
}
