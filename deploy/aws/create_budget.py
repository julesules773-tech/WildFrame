#!/usr/bin/env python3
"""Create the WildFrame monthly AWS cost budget with email alerts.

Standard budgets are free (no per-budget fee). Alerts are sent directly to
the subscriber email — no SNS verification needed.

Usage:
    .venv/bin/python deploy/aws/create_budget.py

Reads AWS credentials from the repo's .env (dotenv), same as the app.
"""
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import boto3

EMAIL = "julesules773@gmail.com"
BUDGET_NAME = "WildFrame monthly"
AMOUNT = "15.0"      # USD per month
WARN_PCT = 80.0      # alert at $12
CRIT_PCT = 100.0     # alert at the ceiling


def _notifications() -> list[dict]:
    return [
        {
            "Notification": {
                "NotificationType": "ACTUAL",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": WARN_PCT,
                "ThresholdType": "PERCENTAGE",
            },
            "Subscribers": [{"SubscriptionType": "EMAIL", "Address": EMAIL}],
        },
        {
            "Notification": {
                "NotificationType": "ACTUAL",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": CRIT_PCT,
                "ThresholdType": "PERCENTAGE",
            },
            "Subscribers": [{"SubscriptionType": "EMAIL", "Address": EMAIL}],
        },
    ]


def main() -> int:
    sts = boto3.client("sts", region_name="us-east-1")
    acct = sts.get_caller_identity()["Account"]
    budgets = boto3.client("budgets", region_name="us-east-1")

    existing = budgets.describe_budgets(AccountId=acct).get("Budgets", [])
    for x in existing:
        nots = budgets.describe_notifications_for_budget(
            AccountId=acct, BudgetName=x["BudgetName"]
        ).get("Notifications", [])
        print(
            f"existing: {x['BudgetName']!r} (limit ${x['BudgetLimit']['Amount']}/mo, "
            f"{len(nots)} notification(s))"
        )

    if any(x["BudgetName"] == BUDGET_NAME for x in existing):
        print(f"budget {BUDGET_NAME!r} already exists — nothing to do")
        return 0

    budgets.create_budget(
        AccountId=acct,
        Budget={
            "BudgetName": BUDGET_NAME,
            "BudgetLimit": {"Amount": AMOUNT, "Unit": "USD"},
            "TimeUnit": "MONTHLY",
            "BudgetType": "COST",
        },
        NotificationsWithSubscribers=_notifications(),
    )
    print(
        f"created {BUDGET_NAME!r}: ${AMOUNT}/mo, ACTUAL alerts at "
        f"{WARN_PCT:.0f}% and {CRIT_PCT:.0f}% -> {EMAIL}"
    )

    ver = budgets.describe_notifications_for_budget(
        AccountId=acct, BudgetName=BUDGET_NAME
    ).get("Notifications", [])
    for n in ver:
        subs = [s["Address"] for s in n.get("Subscribers", [])]
        print(
            f"verified: {n['NotificationType']} > {n['Threshold']:.0f}% "
            f"({n['ComparisonOperator']}) -> {subs}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
