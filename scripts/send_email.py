"""Send an email notification via SMTP. Credentials read from env vars.

Required env vars:
    SMTP_HOST, SMTP_USER, SMTP_PASSWORD, SMTP_TO
Optional:
    SMTP_PORT (default 465), SMTP_FROM (default SMTP_USER)

Usage:
    uv run python scripts/send_email.py --subject "Run done" --body "All good"
    uv run python scripts/send_email.py --status FINISHED --experiment gpt2_124m

Chain after training:
    uv run python scripts/train.py --config configs/gpt2_124m.yaml && \
    uv run python scripts/send_email.py --status FINISHED --experiment gpt2_124m || \
    uv run python scripts/send_email.py --status FAILED --experiment gpt2_124m
"""

import argparse
import os
import smtplib
import socket
import sys
from email.message import EmailMessage
from pathlib import Path


def load_dotenv(path: Path) -> None:
    """Populate os.environ from a KEY=VALUE file. Existing env vars take precedence."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def send_email(subject: str, body: str) -> bool:
    host = os.environ.get("SMTP_HOST")
    port = int(os.environ.get("SMTP_PORT", "465"))
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    to_addr = os.environ.get("SMTP_TO")
    from_addr = os.environ.get("SMTP_FROM", user)

    missing = [
        k
        for k, v in {
            "SMTP_HOST": host,
            "SMTP_USER": user,
            "SMTP_PASSWORD": password,
            "SMTP_TO": to_addr,
        }.items()
        if not v
    ]
    if missing:
        print(f"[send_email] Missing env vars: {', '.join(missing)}", file=sys.stderr)
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(body)

    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=30) as s:
                s.login(user, password)
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=30) as s:
                s.starttls()
                s.login(user, password)
                s.send_message(msg)
        print(f"[send_email] Sent to {to_addr}")
        return True
    except Exception as e:
        print(f"[send_email] Failed: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Send an email notification via SMTP.")
    parser.add_argument(
        "--subject", help="Email subject (overrides --status/--experiment)."
    )
    parser.add_argument("--body", default="", help="Email body.")
    parser.add_argument("--status", help="Run status, e.g. FINISHED or FAILED.")
    parser.add_argument(
        "--experiment", help="Experiment name, included in default subject/body."
    )
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    hostname = socket.gethostname()
    if args.subject:
        subject = args.subject
        body = args.body
    else:
        status = args.status or "DONE"
        exp = args.experiment or "(no experiment)"
        subject = f"[pretrain][{hostname}] {status} — {exp}"
        body = args.body or f"Host: {hostname}\nExperiment: {exp}\nStatus: {status}\n"

    ok = send_email(subject, body)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
