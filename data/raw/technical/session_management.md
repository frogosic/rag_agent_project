# Session Management

This document describes how user sessions are established, maintained, refreshed, and terminated. It is distinct from the authentication API reference: the API reference describes the wire protocol, while this document describes the session lifecycle and the operational concerns around it.

## Session model

A session is the period between a successful login and an explicit logout or session expiration. Each session is backed by a refresh token stored in the auth database, plus an access token held by the client. The session is identified server-side by a `session_id` that is embedded in the refresh token and persisted in the database.

A user may hold multiple concurrent sessions — for example, one on a desktop browser and one on a mobile device. Each session is independent: revoking one does not affect the others. The admin console exposes a per-user view of active sessions and supports targeted revocation.

## Refresh flow

When an access token expires, the client exchanges the refresh token for a new access token via `POST /auth/refresh`:

```json
{
  "refresh_token": "rt_..."
}
```

The successful response includes a new access token and a new refresh token. **The refresh token is rotated on every refresh.** The old refresh token is invalidated as soon as the new one is issued. This is a deliberate security choice: refresh token rotation limits the blast radius of a leaked refresh token to a single use.

If a refresh request arrives with a refresh token that has already been used, the service treats this as a potential token theft scenario and revokes the entire session — not just the presented token. The client receives a `refresh_token_reused` error and must prompt the user to log in again. This policy assumes that the legitimate client will have stored the rotated refresh token; receiving the old one indicates either client misbehavior or a token leak.

## Session expiration

A session ends in one of four ways:

1. **Explicit logout.** The client calls `/auth/logout` and the session is revoked immediately.
2. **Refresh token expiration.** The refresh token reaches its TTL and the next refresh attempt fails. The client must prompt the user to log in again.
3. **Inactivity timeout.** Even if the refresh token is still within its TTL, a session with no activity for the configured inactivity window is invalidated. This is enforced by tracking `last_seen_at` on the session record and rejecting refresh attempts where the gap exceeds the inactivity timeout.
4. **Administrative revocation.** A privileged user revokes the session from the admin console, or the user themselves revokes it from the security settings page.

The inactivity timeout is independent of the refresh token TTL. A session with a 30-day refresh token TTL and a 24-hour inactivity timeout will be invalidated after 24 hours of inactivity, even though the refresh token has 29 more days of nominal validity. This is intentional: the inactivity timeout protects against abandoned sessions on shared devices.

## Token revocation

Revocation is implemented as a denylist in Redis, keyed by token JTI (JWT ID). When a token is revoked, its JTI is added to the denylist with a TTL equal to the token's remaining lifetime. The validation middleware checks the denylist on every request.

The denylist is the reason access tokens are short-lived: a 15-minute access token bounds the worst-case revocation latency to 15 minutes if the denylist is unavailable. If Redis is unreachable, the service fails open for access token validation — accepting tokens that may have been revoked — rather than failing closed and locking out all users. This is a deliberate availability choice and is documented as a known tradeoff.

Refresh tokens are checked against the database (not Redis) on every refresh, so refresh token revocation is immediate and does not depend on cache availability.
