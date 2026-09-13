# app/services/jobs/discovery.py
from jobspy import scrape_jobs


def discover_jobs(role: str, location: str, limit: int = 50) -> list[dict]:
    df = scrape_jobs(
        site_name=["linkedin", "indeed", "glassdoor", "google"],
        search_term=role,
        location=location,
        results_wanted=limit,
        hours_old=0.1,
        country="germany",
    )
    return df.to_dict(orient="records")
