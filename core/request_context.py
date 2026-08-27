"""Request-scoped tenant identity, normally populated by an auth gateway."""

from contextvars import ContextVar, Token
from dataclasses import dataclass


@dataclass(frozen=True)
class RequestIdentity:
    tenant_id: str = "default"
    user_id: str = "anonymous"


_identity: ContextVar[RequestIdentity] = ContextVar(
    "request_identity", default=RequestIdentity()
)


def get_request_identity() -> RequestIdentity:
    return _identity.get()


def set_request_identity(tenant_id: str, user_id: str) -> Token:
    return _identity.set(RequestIdentity(tenant_id=tenant_id, user_id=user_id))


def reset_request_identity(token: Token) -> None:
    _identity.reset(token)
