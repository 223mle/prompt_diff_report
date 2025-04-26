from __future__ import annotations


class ProviderError(RuntimeError):
    """基底例外 - あらゆるプロバイダ関連エラーの親。"""


class ProviderRateLimitError(ProviderError):
    """HTTP 429 などレートリミット超過。"""


class ProviderAuthError(ProviderError):
    """HTTP 401 / 認証失敗。"""
