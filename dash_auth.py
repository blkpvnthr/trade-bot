"""
dash_auth.py — multi-method sign-in for the dashboard: Google + GitHub OAuth + email/password.

Security model: every method only grants a session if the resolved email is on the allow-list
(DASH_ALLOWED_EMAILS). Fully fail-safe — a provider with no credentials is simply hidden, Authlib
being absent disables OAuth (not the dashboard), and the legacy single-password login keeps working.

.env config:
    DASH_ALLOWED_EMAILS=contact@quantumquant.trade        # comma-sep allow-list (REQUIRED for OAuth)
    DASH_USERS=contact@quantumquant.trade:pbkdf2:sha256... # email:hash,... (scripts/hash_password.py)
    GOOGLE_CLIENT_ID=... / GOOGLE_CLIENT_SECRET=...        # Google Cloud OAuth 2.0 client
    GITHUB_CLIENT_ID=... / GITHUB_CLIENT_SECRET=...        # GitHub OAuth App
Redirect URIs to register with each provider:
    https://tv.quantumquant.trade/dash/auth/google/callback
    https://tv.quantumquant.trade/dash/auth/github/callback
"""
import os

from flask import redirect, request, session, url_for
from werkzeug.security import check_password_hash

ALLOWED = {e.strip().lower() for e in
           os.environ.get("DASH_ALLOWED_EMAILS", "contact@quantumquant.trade").split(",") if e.strip()}
GOOGLE_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GITHUB_ID = os.environ.get("GITHUB_CLIENT_ID", "")
GITHUB_SECRET = os.environ.get("GITHUB_CLIENT_SECRET", "")

_oauth = None


def _load_users():
    """email -> password hash, from DASH_USERS='email:hash,email2:hash2'."""
    users = {}
    for pair in os.environ.get("DASH_USERS", "").split(","):
        if ":" in pair:
            email, h = pair.split(":", 1)
            users[email.strip().lower()] = h.strip()
    return users


def providers():
    """Which OAuth providers are configured — drives the login page buttons."""
    return {"google": bool(GOOGLE_ID and GOOGLE_SECRET),
            "github": bool(GITHUB_ID and GITHUB_SECRET),
            "email": bool(_load_users())}


def _grant(email):
    """Grant a session iff the email is on the allow-list."""
    email = (email or "").lower()
    if not email or (ALLOWED and email not in ALLOWED):
        return False
    session.permanent = True
    session["auth"] = True
    session["user"] = email
    return True


def email_login(email, password):
    """Verify email + password against DASH_USERS (hashed). Returns True on success."""
    h = _load_users().get((email or "").lower())
    try:
        return bool(h) and check_password_hash(h, password or "") and _grant(email)
    except Exception:
        return False


def _init_oauth(app):
    global _oauth
    try:
        from authlib.integrations.flask_client import OAuth
    except Exception:
        return None
    o = OAuth(app)
    if GOOGLE_ID and GOOGLE_SECRET:
        o.register("google", client_id=GOOGLE_ID, client_secret=GOOGLE_SECRET,
                   server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
                   client_kwargs={"scope": "openid email profile"})
    if GITHUB_ID and GITHUB_SECRET:
        o.register("github", client_id=GITHUB_ID, client_secret=GITHUB_SECRET,
                   access_token_url="https://github.com/login/oauth/access_token",
                   authorize_url="https://github.com/login/oauth/authorize",
                   api_base_url="https://api.github.com/",
                   client_kwargs={"scope": "user:email"})
    _oauth = o
    return o


def register_auth_routes(app, prefix):
    """Register the OAuth start/callback routes on the Flask app. Fail-safe."""
    _init_oauth(app)

    @app.route(prefix + "/auth/<provider>")
    def oauth_start(provider):
        if _oauth is None or provider not in ("google", "github") or not hasattr(_oauth, provider):
            return redirect(prefix + "/login?err=oauth_unavailable")
        client = getattr(_oauth, provider)
        # Deterministic redirect URI (must EXACTLY match the one registered with the provider).
        # DASH_BASE_URL avoids depending on nginx forwarding Host/X-Forwarded-Proto correctly.
        base = os.environ.get("DASH_BASE_URL", "").rstrip("/")
        redirect_uri = (base + prefix + "/auth/" + provider + "/callback") if base \
            else url_for("oauth_callback", provider=provider, _external=True)
        return client.authorize_redirect(redirect_uri)

    @app.route(prefix + "/auth/<provider>/callback")
    def oauth_callback(provider):
        try:
            client = getattr(_oauth, provider)
            token = client.authorize_access_token()
            email = None
            if provider == "google":
                email = (token.get("userinfo") or {}).get("email")
            elif provider == "github":
                resp = client.get("user/emails", token=token).json()
                primary = next((e for e in resp if e.get("primary") and e.get("verified")), None)
                email = (primary or (resp[0] if resp else {})).get("email")
            if email and _grant(email):
                return redirect(prefix + "/")
            return redirect(prefix + "/login?err=not_allowed")
        except Exception:
            return redirect(prefix + "/login?err=oauth")
