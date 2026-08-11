"""
Daily Zerodha authentication helper.

Run:
    python login.py

Workflow:
1. Opens official Kite login URL.
2. You log in normally.
3. Paste the resulting redirect URL (or request_token) into the terminal.
4. Script exchanges request_token for access_token.
5. KITE_ACCESS_TOKEN is saved into .env.

The access token is valid for the trading day.
"""

import os
import sys
import webbrowser
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv, set_key
from kiteconnect import KiteConnect

ENV_FILE = ".env"

load_dotenv(ENV_FILE)

api_key = os.getenv("KITE_API_KEY", "").strip()
api_secret = os.getenv("KITE_API_SECRET", "").strip()

if not api_key or not api_secret:
    raise SystemExit(
        "Set KITE_API_KEY and KITE_API_SECRET in .env first."
    )

kite = KiteConnect(api_key=api_key)

url = kite.login_url()

print("\nOpening official Zerodha login URL:")
print(url)
webbrowser.open(url)

value = input(
    "\nAfter successful login, paste the FULL redirect URL "
    "from your browser address bar, or only request_token:\n> "
).strip()

if "request_token=" in value:
    parsed = urlparse(value)
    request_token = parse_qs(parsed.query).get(
        "request_token",
        [None],
    )[0]
else:
    request_token = value

if not request_token:
    raise SystemExit("Could not extract request_token.")

try:
    session = kite.generate_session(
        request_token,
        api_secret=api_secret,
    )
except Exception as exc:
    print(f"\n❌ Token exchange failed: {exc}")
    sys.exit(1)

access_token = session["access_token"]

set_key(
    ENV_FILE,
    "KITE_ACCESS_TOKEN",
    access_token,
)

kite.set_access_token(access_token)

profile = kite.profile()

print("\n✅ Zerodha login successful.")
print(
    "User:",
    profile.get(
        "user_name",
        profile.get("user_id"),
    ),
)
print("✅ KITE_ACCESS_TOKEN saved to .env")
print("\nNow run:")
print("python test_connections.py")
print("python bot.py")
