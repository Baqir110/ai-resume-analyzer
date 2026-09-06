# Fail closed on CV fact validation

German Minimal ATS generation validates normalized source career facts—companies, job titles, dates, and degrees—against the generated LaTeX. When a required fact is missing or altered, the API returns HTTP 422 and produces no PDF. We prefer a visible failed request to a polished document containing hallucinated or changed career information.

## Considered Options

- Best-effort generation with prompt-only preservation rules.
- Generate the PDF and warn about facts that could not be verified.
- Reject unverified output before compilation.

## Consequences

Generation may require the user to provide a résumé whose required facts can be extracted unambiguously. This is an intentional safety cost that protects candidate accuracy.
