#!/usr/bin/env python3
"""CLI operator utility for provisioning ModelGuard users with role-based access control.

Usage:
    python backend/scripts/create_user.py --username operator1 --password "SecurePass123!" --role OPERATOR
"""

import argparse
import sys

from app.core.security import get_password_hash
from app.db import models
from app.db.session import SessionLocal, init_db


def main() -> None:
    parser = argparse.ArgumentParser(description="Provision a ModelGuard user with role-based permissions.")
    parser.add_argument("--username", required=True, help="Unique username identifier")
    parser.add_argument("--password", required=True, help="User password (will be hashed with Argon2id)")
    parser.add_argument(
        "--role",
        required=True,
        choices=["VIEWER", "OPERATOR", "APPROVER", "ADMIN"],
        help="Access role to assign (VIEWER, OPERATOR, APPROVER, ADMIN)",
    )

    args = parser.parse_args()

    init_db()
    db = SessionLocal()
    try:
        existing = db.query(models.User).filter(models.User.username == args.username).first()
        if existing:
            print(f"Error: User '{args.username}' already exists (current role: {existing.role}).", file=sys.stderr)
            sys.exit(1)

        hashed = get_password_hash(args.password)
        new_user = models.User(
            username=args.username,
            hashed_password=hashed,
            role=args.role,
            is_active=True,
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        print(f"Successfully provisioned user '{new_user.username}' with role '{new_user.role}' (ID: {new_user.id}).")
    finally:
        db.close()


if __name__ == "__main__":
    main()
