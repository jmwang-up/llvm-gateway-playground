class GatewayError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int,
        retryable: bool,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable

