"""Errors presented by the CLI."""


class GcpError(Exception):
    def __init__(self, message: str, details: str | None = None):
        super().__init__(f"{message}: {details}" if details else message)


def redact(message: str, *secrets: str | None) -> str:
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[redacted]")
    return message
