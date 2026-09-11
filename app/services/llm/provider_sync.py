"""Quota synchronization utilities for multi-provider setup."""
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class ProviderQuotaSyncService:
    """Service to periodically sync provider quotas."""
    
    def __init__(
        self,
        sync_interval_minutes: int = 5,
        enable_sync: bool = True,
    ):
        self.sync_interval_minutes = sync_interval_minutes
        self.enable_sync = enable_sync
        self.last_sync_time = {}
        self.sync_history = []
    
    async def background_sync_quotas(
        self,
        quota_manager: Any,
        providers: Dict[str, Any],
    ) -> None:
        """Run quota sync in background at regular intervals.
        
        Args:
            quota_manager: QuotaSyncManager instance
            providers: Dict of provider clients
        """
        if not self.enable_sync:
            return
        
        while True:
            try:
                await asyncio.sleep(self.sync_interval_minutes * 60)
                
                logger.info("Starting background quota sync...")
                results = await quota_manager.sync_provider_quotas(providers)
                
                self.sync_history.append({
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'results': results,
                })
                
                # Keep only last 100 syncs
                if len(self.sync_history) > 100:
                    self.sync_history = self.sync_history[-100:]
                
                logger.info(f"Quota sync completed: {len(results)} providers checked")
            
            except Exception as e:
                logger.error(f"Background quota sync failed: {e}")
                # Continue running despite errors
    
    def get_sync_history(self, limit: int = 20) -> list:
        """Get recent sync history."""
        return self.sync_history[-limit:]
    
    def get_latest_sync(self, provider: Optional[str] = None) -> Optional[Dict]:
        """Get latest sync result for provider."""
        if not self.sync_history:
            return None
        
        latest = self.sync_history[-1]
        
        if provider:
            return latest.get('results', {}).get(provider)
        
        return latest
