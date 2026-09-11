"""Enhanced LLM provider with retry logic and timeout handling."""
import asyncio
import logging
from typing import Optional, Any, Callable
from datetime import datetime, timedelta
import random
from functools import wraps

import aiohttp
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

logger = logging.getLogger(__name__)


class ProviderTimeoutError(Exception):
    """Raised when a provider call times out."""
    pass


class ProviderRetryExhausted(Exception):
    """Raised when all retry attempts are exhausted."""
    pass


class ProviderRetryConfig:
    """Configuration for provider retry behavior."""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        timeout_seconds: float = 30.0,
        jitter: bool = True,
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.timeout_seconds = timeout_seconds
        self.jitter = jitter


def exponential_backoff_with_jitter(
    attempt: int,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
) -> float:
    """Calculate delay with exponential backoff and optional jitter.
    
    Args:
        attempt: Current attempt number (0-indexed)
        initial_delay: Base delay in seconds
        max_delay: Maximum delay cap
        exponential_base: Exponential multiplier
        jitter: Whether to add random jitter
    
    Returns:
        Delay in seconds before next retry
    """
    delay = min(initial_delay * (exponential_base ** attempt), max_delay)
    
    if jitter:
        # Add random jitter: ±10% of calculated delay
        jitter_amount = delay * 0.1
        delay += random.uniform(-jitter_amount, jitter_amount)
    
    return max(0, delay)


class ProviderWithRetry:
    """Wrapper for LLM provider calls with automatic retry logic."""
    
    def __init__(
        self,
        provider_name: str,
        config: Optional[ProviderRetryConfig] = None,
    ):
        self.provider_name = provider_name
        self.config = config or ProviderRetryConfig()
        self.retry_history = []
    
    async def call_with_retry(
        self,
        async_callable: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """Execute an async callable with retry logic.
        
        Args:
            async_callable: Async function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Result from the callable
        
        Raises:
            ProviderRetryExhausted: If all retries fail
            ProviderTimeoutError: If timeout occurs
        """
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Set timeout for the call
                return await asyncio.wait_for(
                    async_callable(*args, **kwargs),
                    timeout=self.config.timeout_seconds,
                )
            
            except asyncio.TimeoutError as e:
                last_exception = ProviderTimeoutError(
                    f"{self.provider_name} call timed out after "
                    f"{self.config.timeout_seconds}s"
                )
                self._log_retry_attempt(attempt, last_exception, timed_out=True)
            
            except (aiohttp.ClientError, ConnectionError) as e:
                last_exception = e
                self._log_retry_attempt(attempt, e)
            
            except Exception as e:
                # Don't retry on unexpected errors
                logger.error(
                    f"{self.provider_name} call failed with unexpected error: {e}"
                )
                raise
            
            # If this was the last attempt, raise
            if attempt >= self.config.max_retries:
                break
            
            # Calculate backoff delay
            delay = exponential_backoff_with_jitter(
                attempt,
                self.config.initial_delay,
                self.config.max_delay,
                jitter=self.config.jitter,
            )
            
            logger.warning(
                f"{self.provider_name}: Attempt {attempt + 1} failed, "
                f"retrying in {delay:.2f}s (error: {last_exception})"
            )
            
            await asyncio.sleep(delay)
        
        # All retries exhausted
        raise ProviderRetryExhausted(
            f"{self.provider_name}: All {self.config.max_retries} retries exhausted. "
            f"Last error: {last_exception}"
        )
    
    def _log_retry_attempt(
        self,
        attempt: int,
        exception: Exception,
        timed_out: bool = False,
    ) -> None:
        """Log retry attempt details."""
        self.retry_history.append({
            'attempt': attempt,
            'timestamp': datetime.utcnow().isoformat(),
            'provider': self.provider_name,
            'error': str(exception),
            'timed_out': timed_out,
        })
        
        logger.debug(
            f"{self.provider_name} retry #{attempt}: {'TIMEOUT' if timed_out else 'ERROR'} - {exception}"
        )
    
    def get_retry_history(self) -> list:
        """Return retry history for this provider."""
        return self.retry_history
    
    def clear_retry_history(self) -> None:
        """Clear retry history."""
        self.retry_history.clear()
