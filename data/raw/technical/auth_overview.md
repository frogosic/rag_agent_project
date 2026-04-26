# Authentication Overview

This document describes the authentication model used across our services. It covers the conceptual model and decision guidance for choosing an auth strategy. For specific endpoints, see the API reference. For deployment configuration, see the configuration guide. For session and refresh flows, see session management.

## Authentication vs Authorization

Authentication answers "who are you," authorization answers "what are you allowed to do." This document covers authentication only. Authorization is enforced at the resource layer via role-based access control and is described in the access control guide.

## Supported authentication methods

We support three authentication methods, each suited to different use cases.

**Bearer tokens** are the default for first-party clients. The client obtains a token by exchanging credentials and includes the token in the `Authorization` header on subsequent requests. Bearer tokens are short-lived and paired with refresh tokens for session continuity.

**API keys** are used for server-to-server integrations and third-party access. API keys are long-lived, scoped to a specific integration, and revocable from the admin console. API keys do not support refresh — when an API key is rotated, the integration must be updated.

**OAuth 2.0 authorization code flow** is used when a third-party application needs to act on behalf of a user. The third-party redirects the user to our authorization endpoint, the user grants consent, and the third-party receives a code that is exchanged for a token.

## When to use which method

Choose bearer tokens for any first-party client (web app, mobile app, internal tools). Choose API keys when a backend service needs to call our API without a user context — for example, a billing reconciliation job. Choose OAuth when you are building an integration that other developers will use to access user data.

Bearer tokens and OAuth tokens flow through the same validation path on our side; the difference is how they are obtained. API keys take a separate validation path because they do not expire and do not have an associated user session.

## Failure modes

Authentication can fail for several conceptual reasons: the credential is invalid, the credential is expired, the credential is revoked, or the credential is valid but the request is malformed. The API reference enumerates the specific error codes returned for each case. Conceptually, clients should treat all authentication failures as terminal for the current request and re-authenticate before retrying — retrying with the same failed credential will not succeed.

Rate limiting is layered on top of authentication. A client whose credential is valid but whose request rate exceeds the limit will receive a rate-limit error rather than an authentication error. Clients should distinguish between the two and back off appropriately on rate-limit errors rather than re-authenticating.
