# Authentication API Reference

This document specifies the authentication endpoints, request and response shapes, and error codes. For conceptual guidance on choosing an auth method, see the authentication overview. For session and refresh flows, see session management.

## POST /auth/login

Exchange user credentials for an access token and refresh token.

Request body:

```json
{
  "email": "user@example.com",
  "password": "..."
}
```

Successful response (200):

```json
{
  "access_token": "eyJ...",
  "refresh_token": "rt_...",
  "token_type": "Bearer",
  "expires_in": 900
}
```

The `expires_in` field is the access token lifetime in seconds. The refresh token lifetime is not returned in the response and must be inferred from the configuration documentation.

## POST /auth/token

Exchange an OAuth authorization code for a token. Used in the OAuth 2.0 authorization code flow.

Request body:

```json
{
  "grant_type": "authorization_code",
  "code": "...",
  "client_id": "...",
  "client_secret": "...",
  "redirect_uri": "..."
}
```

This endpoint is only used for OAuth flows. First-party clients should use `/auth/login` instead.

## POST /auth/logout

Revoke the current access token and any associated refresh token. Requires a valid bearer token in the `Authorization` header. Returns 204 on success. Logout is idempotent — calling it with an already-revoked token returns 204, not an error.

## GET /auth/me

Return the authenticated user's profile. Requires a valid bearer token. Returns 200 with the user object, or 401 if the token is invalid or expired.

## Error codes

Authentication endpoints return errors in a uniform shape:

```json
{
  "error": "invalid_credentials",
  "error_description": "Email or password is incorrect.",
  "request_id": "req_..."
}
```

The error codes returned by authentication endpoints are:

- `invalid_credentials` — the email or password did not match. Returned by `/auth/login` only. HTTP 401.
- `token_expired` — the access token has passed its expiration time. Returned by any authenticated endpoint. HTTP 401.
- `token_revoked` — the token was valid but has been explicitly revoked, typically via logout or admin action. HTTP 401.
- `token_malformed` — the token does not parse as a valid JWT. HTTP 400.
- `insufficient_scope` — the token is valid but lacks the scope required for the requested resource. HTTP 403.
- `rate_limited` — the client has exceeded the rate limit for this endpoint. HTTP 429. Includes a `Retry-After` header.

Clients should match on `error` rather than parsing `error_description`, which is human-readable and may change between versions.
