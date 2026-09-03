#!/usr/bin/env python3
"""Generate a DASH_USERS entry for dashboard email/password login.

    python3 scripts/hash_password.py you@email.com

Prompts for a password (hidden) and prints  email:hash  to paste into DASH_USERS in .env
(comma-separate multiple users). The plaintext password is never stored.
"""
import getpass
import sys

from werkzeug.security import generate_password_hash

email = (sys.argv[1] if len(sys.argv) > 1 else input("email: ")).strip().lower()
pw = getpass.getpass("password: ")
if getpass.getpass("confirm:  ") != pw:
    print("passwords don't match")
    sys.exit(1)
if len(pw) < 8:
    print("use at least 8 characters")
    sys.exit(1)
print("\nAdd to DASH_USERS in .env (comma-separate multiple users):\n")
print(f"{email}:{generate_password_hash(pw)}")
