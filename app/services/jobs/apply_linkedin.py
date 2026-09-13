# app/services/jobs/apply_linkedin.py
from playwright.async_api import async_playwright


async def apply_easy_apply(job_url: str, profile: dict, pdf_path: str):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            storage_state="linkedin_session.json",
            user_agent="Mozilla/5.0 ...",
        )
        page = await context.new_page()
        await page.goto(job_url)
        await page.click("button.jobs-apply-button")
        # ... fill fields, upload resume, answer questions
