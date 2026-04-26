# Authentication Configuration

This document describes the deployment-time configuration for the authentication service. For endpoint specifications, see the API reference. For conceptual guidance, see the authentication overview.

## Environment variables

The authentication service reads the following environment variables at startup. All variables are required unless marked optional; the service will refuse to start if a required variable is missing.

- `AUTH_JWT_SIGNING_KEY` — the secret used to sign access tokens. Must be at least 32 bytes. Rotated quarterly.
- `AUTH_JWT_ALGORITHM` — defaults to `HS256`. Set to `RS256` for asymmetric signing in production.
- `AUTH_ACCESS_TOKEN_TTL` — access token lifetime in seconds. Default `900` (15 minutes).
- `AUTH_REFRESH_TOKEN_TTL` — refresh token lifetime in seconds. Default `2592000` (30 days).
- `AUTH_SESSION_INACTIVITY_TIMEOUT` — seconds of inactivity after which a session is invalidated even if the refresh token is still within its TTL. Default `86400` (24 hours).
- `AUTH_DB_URL` — connection string for the auth database, which stores refresh tokens, sessions, and revocation records.
- `AUTH_REDIS_URL` — connection string for the Redis instance used for the token revocation cache.
- `AUTH_RATE_LIMIT_PER_MINUTE` — per-IP rate limit on `/auth/login`. Default `10`.
- `AUTH_OAUTH_CLIENT_REGISTRY_URL` — optional, used only when OAuth is enabled. Points to the service that maintains the registry of authorized OAuth clients.

## Token TTL policy

Access tokens are short-lived (15 minutes by default) and refresh tokens are long-lived (30 days by default). The 15-minute access token TTL is a deliberate tradeoff: short enough that revocation latency is bounded without consulting the revocation cache on every request, long enough that the refresh endpoint is not hammered. Do not raise this value above 3600 (1 hour) without security review.

The refresh token TTL of 30 days reflects the maximum acceptable session duration before forced re-authentication. Reducing it below 7 days has been shown to materially hurt user experience in our deployment; raising it above 30 days requires security review.

## Key rotation

JWT signing keys are rotated on a quarterly schedule. During rotation, both the previous and current keys are accepted for verification for a 24-hour overlap window, after which only the new key is accepted. Rotation is performed by updating the `AUTH_JWT_SIGNING_KEY` environment variable and triggering a rolling restart; the service reads both `AUTH_JWT_SIGNING_KEY` and `AUTH_JWT_SIGNING_KEY_PREVIOUS` during the overlap window.

API keys are not rotated on a schedule. They are rotated reactively — when an integration's keys are suspected of being compromised, or when an integration is decommissioned. API key rotation requires the integration to update its stored key; there is no overlap window for API keys.

## Per-environment configuration

The default values above apply to production. Staging uses the same defaults with one exception: `AUTH_RATE_LIMIT_PER_MINUTE` is set to `100` in staging to permit load testing. Development environments use `AUTH_ACCESS_TOKEN_TTL=60` to make expiration scenarios easier to test, and disable rate limiting entirely by setting `AUTH_RATE_LIMIT_PER_MINUTE=0`.
