#!/usr/bin/env python3
"""Create a local dev user for the investor/government dashboard.

Skips the invite-token flow entirely — just creates the user directly.
Safe to re-run: exits cleanly if the email already exists.

Usage:
    .venv/bin/python create_user.py                           # investor (default)
    .venv/bin/python create_user.py --role government         # government role
    .venv/bin/python create_user.py --email me@test.com --name "Julian" --role investor
    .venv/bin/python create_user.py --admin                   # internal (admin) role
"""

import argparse
import os
import sys

from dotenv import load_dotenv
load_dotenv()

from werkzeug.security import generate_password_hash
import db


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--email", default="admin@wildframe.local")
    parser.add_argument("--name", default="Admin")
    parser.add_argument("--password", default="password123")
    parser.add_argument("--role", default="investor",
                        choices=["investor", "government", "internal"])
    parser.add_argument("--admin", action="store_true",
                        help="shortcut for --role internal")
    args = parser.parse_args()

    if args.admin:
        args.role = "internal"

    existing = db.get_user_by_email(args.email)
    if existing:
        print(f"User already exists: {args.email} (role={existing['role']}, id={existing['id']})")
        print(f"  → Login at http://localhost:4141/login")
        return 0

    pw_hash = generate_password_hash(args.password, method="pbkdf2:sha256", salt_length=16)
    user = db.create_user(args.email, pw_hash, args.name, role=args.role)

    print(f"✅ Created user:")
    print(f"   email:    {user['email']}")
    print(f"   name:     {user['name']}")
    print(f"   role:     {user['role']}")
    print(f"   password: {args.password}")
    print(f"   id:       {user['id']}")
    print()
    print(f"   → Login at http://localhost:4141/login")
    print(f"   → Dashboard at http://localhost:4141/dashboard")
    return 0


if __name__ == "__main__":
    sys.exit(main())
