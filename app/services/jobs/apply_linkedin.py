# app/services/jobs/apply_linkedin.py
"""LinkedIn Easy Apply -- deliberately not implemented.

This used to be a half-finished Playwright stub (hardcoded visible
browser, a `linkedin_session.json` cookie file the user would have had
to produce by hand, no real field-filling logic). Rather than complete
it, this has been intentionally left unimplemented:

LinkedIn's User Agreement prohibits automated interaction with the
platform beyond their own official API, and LinkedIn actively detects
and bans automation on accounts, including scripted Easy Apply flows.
That's a materially different risk profile from the generic
browser-use pipeline in browser_use_applier.py, which navigates
whatever job_url it's given and already stops cleanly (no bypass) the
moment a site's anti-bot measures kick in, per Section 21/40 -- see
your own agent run log, where it hit Cloudflare on an Indeed URL and
correctly reported failure rather than working around it.

Building a *dedicated* LinkedIn automation path doesn't add meaningful
coverage the generic pipeline doesn't already reach for other sites --
it specifically targets the one platform whose ToS explicitly forbids
this and whose enforcement risks the user's account. That tradeoff
isn't worth it, so this module is a documented no-op instead of a
partially-working bypass.

If LinkedIn jobs come up during discovery, they still flow through the
normal decision engine and browser_use_applier -- they just get the
same generic, non-bypassing treatment as any other site, and stop if
LinkedIn's automation detection kicks in.
"""

from __future__ import annotations


async def apply_easy_apply(job_url: str, profile: dict, pdf_path: str) -> None:
    raise NotImplementedError(
        "LinkedIn Easy Apply automation is intentionally not implemented -- "
        "see the module docstring in apply_linkedin.py for why. Jobs with a "
        "LinkedIn application_url are handled by the generic browser-use "
        "pipeline instead, which stops (rather than bypasses) if LinkedIn's "
        "anti-automation detection triggers."
    )


__all__ = ["apply_easy_apply"]
