"""Notification fan-out: in-app notifications + pluggable email transport."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy.orm import Session

from .config import settings
from .db import Asset, Generation, Notification, User

log = logging.getLogger(__name__)


@dataclass
class Email:
    to: str
    subject: str
    text: str
    html: str | None = None


class EmailTransport(Protocol):
    name: str
    def send(self, email: Email) -> bool: ...


class ConsoleTransport:
    name = "console"

    def send(self, email: Email) -> bool:
        log.info("[email:%s] -> %s | %s\n%s", self.name, email.to, email.subject, email.text)
        print(f"\n--- EMAIL (console transport) ---\nTo: {email.to}\nSubject: {email.subject}\n\n{email.text}\n---\n")
        return True


class ResendTransport:
    name = "resend"

    def send(self, email: Email) -> bool:
        import json
        import urllib.request
        if not settings.resend_api_key:
            log.warning("RESEND_API_KEY missing; dropping email")
            return False
        body = json.dumps({"from": "3DFUN AI <noreply@example.com>", "to": [email.to],
                           "subject": email.subject, "text": email.text,
                           "html": email.html or ""}).encode()
        req = urllib.request.Request("https://api.resend.com/emails", data=body,
                                     headers={"Authorization": f"Bearer {settings.resend_api_key}",
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return 200 <= resp.status < 300
        except Exception as exc:      # pragma: no cover - network
            log.warning("resend delivery failed: %s", exc)
            return False


def transport() -> EmailTransport:
    if settings.email_provider == "resend" and settings.resend_api_key:
        return ResendTransport()
    return ConsoleTransport()


def _shell(title: str, body: str, cta: str = "", url: str = "") -> tuple[str, str]:
    text = f"{title}\n\n{body}\n"
    if cta and url:
        text += f"\n{cta}: {url}\n"
    text += "\n— 3DFUN AI\n"
    html_body = body.replace("\n", "<br>")
    html = (f"<h2 style='font-family:sans-serif'>{title}</h2>"
            f"<p style='font-family:sans-serif'>{html_body}</p>"
            + (f"<p><a href='{url}' style='font-family:sans-serif'>{cta}</a></p>" if cta else "")
            + "<p style='color:#666;font-family:sans-serif;font-size:12px'>3DFUN AI</p>")
    return text, html


def send_email(to: str, subject: str, title: str, body: str, cta: str = "", url: str = "") -> bool:
    text, html = _shell(title, body, cta, url)
    return transport().send(Email(to=to, subject=subject, text=text, html=html))


def notify(db: Session, user_id: str, kind: str, title: str, body: str = "") -> Notification:
    n = Notification(user_id=user_id, kind=kind, title=title, body=body)
    db.add(n)
    db.commit()
    db.refresh(n)
    return n


# ---------------------------------------------------------------------------
# transactional templates
# ---------------------------------------------------------------------------
def send_verify_email(user: User, token: str) -> bool:
    url = f"{settings.app_url}/app/verify-email?token={token}"
    return send_email(user.email, "Verify your 3DFUN AI account",
                      "Verify your email",
                      "Confirm your address to activate generation credits.",
                      "Verify email", url)


def send_password_reset(user: User, token: str) -> bool:
    url = f"{settings.app_url}/app/reset-password?token={token}"
    return send_email(user.email, "Reset your 3DFUN AI password",
                      "Password reset requested",
                      "This link expires in 60 minutes. If you did not request it, ignore this email.",
                      "Reset password", url)


def send_welcome(user: User, credits: int) -> bool:
    return send_email(user.email, "Welcome to 3DFUN AI",
                      "Your 3D studio is ready",
                      f"We added {credits} credits to your account. Upload a photo and generate your first model.",
                      "Open the studio", f"{settings.app_url}/app/studio")


def send_security_alert(user: User, what: str) -> bool:
    return send_email(user.email, f"Security alert: {what}",
                      "Security alert", f"We detected: {what}. If this was not you, reset your password.",
                      "Review sessions", f"{settings.app_url}/app/settings")


def notify_generation_complete(db: Session, gen: Generation, asset: Asset) -> None:
    notify(db, gen.user_id, "generation.completed", "Your 3D model is ready",
           f"{asset.name} · {asset.polygons:,} triangles")
    user = db.get(User, gen.user_id)
    if user:
        send_email(user.email, "Your 3D model is ready",
                   "Generation complete",
                   f"{asset.name} finished in {round((gen.duration_ms or 0)/1000)}s.",
                   "Open the model", f"{settings.app_url}/app/assets/{asset.id}")


def notify_generation_failed(db: Session, gen: Generation) -> None:
    notify(db, gen.user_id, "generation.failed", "Generation failed",
           f"{gen.error_message or 'Unknown error'} (support {gen.support_code})")
