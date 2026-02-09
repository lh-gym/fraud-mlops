#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import smtplib
from email.mime.text import MIMEText
from pathlib import Path
from urllib import request


def send_slack(message: str) -> None:
    webhook_url = os.getenv("SLACK_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return

    payload = json.dumps({"text": message}).encode("utf-8")
    req = request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=10) as response:
        _ = response.read()


def send_email(subject: str, body: str) -> None:
    sender = os.getenv("ALERT_EMAIL_FROM", "").strip()
    recipient = os.getenv("ALERT_EMAIL_TO", "").strip()
    smtp_host = os.getenv("SMTP_HOST", "").strip()
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USERNAME", "").strip()
    smtp_password = os.getenv("SMTP_PASSWORD", "").strip()

    if not sender or not recipient or not smtp_host:
        return

    message = MIMEText(body, "plain", "utf-8")
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = recipient

    with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
        server.starttls()
        if smtp_user and smtp_password:
            server.login(smtp_user, smtp_password)
        server.sendmail(sender, [recipient], message.as_string())


def append_local_log(row: dict) -> None:
    log_path = Path("artifacts/dashboard/notifications.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Send Slack/email notifications for pipeline status.")
    parser.add_argument("--status", required=True, choices=["success", "failure"])
    parser.add_argument("--message", required=True)
    parser.add_argument("--run-id", default=os.getenv("METAFLOW_RUN_ID", "unknown"))
    args = parser.parse_args()

    subject = f"[MLOps Pipeline] {args.status.upper()} run={args.run_id}"
    body = f"{args.message}\nrun_id={args.run_id}"
    formatted = f"{subject}\n{body}"

    send_slack(formatted)
    send_email(subject=subject, body=body)
    append_local_log({"status": args.status, "run_id": args.run_id, "message": args.message})
    print(formatted)


if __name__ == "__main__":
    main()

