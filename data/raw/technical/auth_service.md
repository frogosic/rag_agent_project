# Authentication Service

## Overview
The authentication service handles all user identity verification. It issues JWT tokens with configurable expiry and supports OAuth2 flows.

## Endpoints

### POST /auth/login
Accepts email and password. Returns a signed JWT access token and refresh token.

**Request:**
```json
{ "email": "user@example.com", "password": "secret" }
```

**Response:**
```json
{ "access_token": "...", "refresh_token": "...", "expires_in": 3600 }
```

### POST /auth/refresh
Accepts a valid refresh token. Returns a new access token.

### POST /auth/logout
Invalidates the refresh token server-side.

## Token Structure
Access tokens expire after 1 hour. Refresh tokens expire after 30 days. Both are RS256 signed.

## Configuration
Set `AUTH_SECRET_KEY` and `AUTH_TOKEN_EXPIRY` in environment variables.

## Error Codes
- `401` — invalid credentials
- `403` — token expired or revoked
- `429` — rate limited, max 10 login attempts per minute per IP
