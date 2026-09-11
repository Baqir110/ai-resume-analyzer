"""Enhanced quota tracking with provider API synchronization."""
import logging
from typing import Dict, Optional, List, Any
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class QuotaSyncManager:
    """Manages synchronization between local log and provider APIs."""
    
    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.last_sync = {}
        self.sync_interval = timedelta(minutes=5)  # Min time between syncs
    
    async def sync_provider_quotas(
        self,
        providers: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Synchronize quota data with provider APIs.
        
        Args:
            providers: Dict mapping provider names to their client objects
        
        Returns:
            Dict of provider quota data with drift detection
        """
        sync_results = {}
        now = datetime.now(timezone.utc)
        
        for provider_name, provider_client in providers.items():
            # Check if enough time has passed since last sync
            last = self.last_sync.get(provider_name)
            if last and (now - last) < self.sync_interval:
                logger.debug(f"Skipping {provider_name} quota sync (too frequent)")
                continue
            
            try:
                api_quota = await self._fetch_provider_quota(
                    provider_name,
                    provider_client,
                )
                
                local_quota = self._get_local_quota(provider_name)
                
                drift = self._detect_drift(api_quota, local_quota)
                
                sync_results[provider_name] = {
                    'api_quota': api_quota,
                    'local_quota': local_quota,
                    'drift_detected': drift is not None,
                    'drift_details': drift,
                    'synced_at': now.isoformat(),
                }
                
                self.last_sync[provider_name] = now
                
                if drift:
                    logger.warning(
                        f"{provider_name} quota drift detected: {drift}"
                    )
            
            except Exception as e:
                logger.error(f"Failed to sync {provider_name} quota: {e}")
                sync_results[provider_name] = {
                    'error': str(e),
                    'synced_at': now.isoformat(),
                }
        
        return sync_results
    
    async def _fetch_provider_quota(
        self,
        provider_name: str,
        provider_client: Any,
    ) -> Optional[Dict[str, Any]]:
        """Fetch quota info from provider API.
        
        Supports: Google Gemini, OpenAI, Anthropic
        """
        if provider_name == 'google':
            return await self._fetch_google_quota(provider_client)
        elif provider_name == 'openai':
            return await self._fetch_openai_quota(provider_client)
        elif provider_name == 'anthropic':
            return await self._fetch_anthropic_quota(provider_client)
        else:
            logger.debug(f"No quota API available for {provider_name}")
            return None
    
    async def _fetch_google_quota(self, client: Any) -> Optional[Dict[str, Any]]:
        """Fetch quota from Google Gemini API."""
        try:
            # Google doesn't expose remaining quota via API for API-key access
            # Instead, we track from headers
            logger.debug("Google quota tracking limited to response headers")
            return {
                'provider': 'google',
                'method': 'response_headers',
                'note': 'Full quota info requires service account',
            }
        except Exception as e:
            logger.error(f"Failed to fetch Google quota: {e}")
            return None
    
    async def _fetch_openai_quota(self, client: Any) -> Optional[Dict[str, Any]]:
        """Fetch quota from OpenAI API."""
        try:
            # OpenAI's usage API endpoint
            import httpx
            
            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(
                    "https://api.openai.com/v1/usage/quota",
                    headers={
                        'Authorization': f'Bearer {client.api_key}',
                    },
                    timeout=10,
                )
                
                if response.status_code == 200:
                    return response.json()
        
        except Exception as e:
            logger.error(f"Failed to fetch OpenAI quota: {e}")
        
        return None
    
    async def _fetch_anthropic_quota(self, client: Any) -> Optional[Dict[str, Any]]:
        """Fetch quota from Anthropic API."""
        # Anthropic doesn't currently expose quota via API
        logger.debug("Anthropic quota API not available")
        return None
    
    def _get_local_quota(self, provider_name: str) -> Dict[str, Any]:
        """Get quota computed from local event log."""
        if not self.log_path.exists():
            return {}
        
        try:
            from app.services.llm.quota_tracker import build_quota_status
            return build_quota_status(
                provider=provider_name,
                log_path=self.log_path,
            )
        except Exception as e:
            logger.error(f"Failed to compute local quota: {e}")
            return {}
    
    def _detect_drift(
        self,
        api_quota: Optional[Dict],
        local_quota: Optional[Dict],
    ) -> Optional[Dict[str, Any]]:
        """Detect discrepancies between API and local quota.
        
        Returns:
            None if no drift, else dict with details
        """
        if not api_quota or not local_quota:
            return None
        
        drift_details = {}
        
        # Compare request counts
        api_requests = api_quota.get('total_requests', 0)
        local_requests = local_quota.get('windows', {}).get('minute', {}).get('used_requests', 0)
        
        if api_requests and local_requests:
            diff = abs(api_requests - local_requests)
            if diff > 5:  # Threshold
                drift_details['request_count_diff'] = diff
        
        return drift_details if drift_details else None
