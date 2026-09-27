"""CV variant routing — classifies a job into a role category and returns
per-variant generation hints consumed by the LaTeX generator and optimizer."""

from __future__ import annotations

import copy
from enum import Enum
from typing import Dict, List

__all__ = ["RoleVariant", "classify_role", "get_variant_config"]


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


class RoleVariant(str, Enum):
    IT_SUPPORT = "it_support"
    IT_ADMIN = "it_admin"
    DEVOPS = "devops"
    MLOPS = "mlops"
    SOFTWARE_ENGINEER = "software_engineer"
    DATA_ENGINEER = "data_engineer"
    CLOUD_ENGINEER = "cloud_engineer"
    GENERIC = "generic"


# ---------------------------------------------------------------------------
# Keyword tables (lower-case; matched against lower-cased title + description)
# ---------------------------------------------------------------------------

_KEYWORDS: Dict[RoleVariant, List[str]] = {
    RoleVariant.DEVOPS: [
        "devops",
        "kubernetes",
        "k8s",
        "docker",
        "ci/cd",
        "pipeline",
        "jenkins",
        "terraform",
        "infrastructure",
        "platform engineer",
        "helm",
        "ansible",
        "gitlab ci",
        "github actions",
        "argocd",
    ],
    RoleVariant.MLOPS: [
        "mlops",
        "ml engineer",
        "machine learning",
        "ai engineer",
        "model deployment",
        "mlflow",
        "kubeflow",
        "data science",
        "feature store",
        "model registry",
        "deep learning",
    ],
    RoleVariant.IT_SUPPORT: [
        "it support",
        "helpdesk",
        "help desk",
        "service desk",
        "1st level",
        "2nd level",
        "ict support",
        "user support",
        "it-support",
        "technical support",
        "desktop support",
        "end user support",
    ],
    RoleVariant.IT_ADMIN: [
        "system administrator",
        "sysadmin",
        "it administrator",
        "network admin",
        "windows admin",
        "linux admin",
        "active directory",
        "group policy",
        "server administration",
        "it operations",
    ],
    RoleVariant.DATA_ENGINEER: [
        "data engineer",
        "data pipeline",
        "etl",
        "spark",
        "kafka",
        "airflow",
        "data warehouse",
        "dbt",
        "flink",
        "hive",
        "data lake",
        "bigquery",
    ],
    RoleVariant.CLOUD_ENGINEER: [
        "cloud engineer",
        "aws engineer",
        "azure engineer",
        "gcp",
        "cloud architect",
        "solutions architect",
        "cloud infrastructure",
        "aws",
        "azure",
        "cloud native",
        "serverless",
    ],
    RoleVariant.SOFTWARE_ENGINEER: [
        "software engineer",
        "backend developer",
        "full stack",
        "fullstack",
        "web developer",
        "python developer",
        "java developer",
        "software developer",
        "backend engineer",
        "api developer",
        "microservices",
    ],
}

# Config table -----------------------------------------------------------

_CONFIGS: Dict[RoleVariant, Dict] = {
    RoleVariant.DEVOPS: {
        "variant": RoleVariant.DEVOPS.value,
        "top_skills": [
            "Kubernetes",
            "Docker",
            "Terraform",
            "CI/CD",
            "Jenkins",
            "Ansible",
            "Helm",
            "GitLab CI",
            "GitHub Actions",
            "Linux",
        ],
        "section_order": ["experience", "skills", "education", "certifications"],
        "cv_title_override": "DevOps Engineer",
        "keyword_boost": [
            "infrastructure as code",
            "container orchestration",
            "deployment pipeline",
            "platform engineering",
        ],
    },
    RoleVariant.MLOPS: {
        "variant": RoleVariant.MLOPS.value,
        "top_skills": [
            "MLflow",
            "Kubeflow",
            "Python",
            "TensorFlow",
            "PyTorch",
            "Docker",
            "Kubernetes",
            "Model Serving",
            "Feature Engineering",
        ],
        "section_order": ["experience", "skills", "projects", "education"],
        "cv_title_override": "MLOps Engineer",
        "keyword_boost": [
            "model deployment",
            "ml pipeline",
            "model monitoring",
            "experiment tracking",
            "feature store",
        ],
    },
    RoleVariant.IT_SUPPORT: {
        "variant": RoleVariant.IT_SUPPORT.value,
        "top_skills": [
            "Windows 10/11",
            "Active Directory",
            "Microsoft 365",
            "ITIL",
            "ServiceNow",
            "Ticketing Systems",
            "Remote Desktop",
        ],
        "section_order": ["experience", "skills", "certifications", "education"],
        "cv_title_override": "IT Specialist",
        "keyword_boost": [
            "incident resolution",
            "user support",
            "service desk",
            "hardware troubleshooting",
            "SLA adherence",
        ],
    },
    RoleVariant.IT_ADMIN: {
        "variant": RoleVariant.IT_ADMIN.value,
        "top_skills": [
            "Windows Server",
            "Active Directory",
            "Group Policy",
            "PowerShell",
            "Linux",
            "VMware",
            "Hyper-V",
            "DNS",
            "DHCP",
        ],
        "section_order": ["experience", "skills", "certifications", "education"],
        "cv_title_override": "IT Systems Administrator",
        "keyword_boost": [
            "server administration",
            "network management",
            "patch management",
            "backup and recovery",
            "identity management",
        ],
    },
    RoleVariant.DATA_ENGINEER: {
        "variant": RoleVariant.DATA_ENGINEER.value,
        "top_skills": [
            "Apache Spark",
            "Kafka",
            "Airflow",
            "Python",
            "SQL",
            "dbt",
            "BigQuery",
            "Snowflake",
            "Databricks",
            "ETL",
        ],
        "section_order": ["experience", "skills", "projects", "education"],
        "cv_title_override": "Data Engineer",
        "keyword_boost": [
            "data pipeline",
            "ETL/ELT",
            "data warehouse",
            "stream processing",
            "data modelling",
        ],
    },
    RoleVariant.CLOUD_ENGINEER: {
        "variant": RoleVariant.CLOUD_ENGINEER.value,
        "top_skills": [
            "AWS",
            "Azure",
            "GCP",
            "Terraform",
            "CloudFormation",
            "Kubernetes",
            "Docker",
            "IAM",
            "VPC",
            "Cloud Security",
        ],
        "section_order": ["experience", "skills", "certifications", "education"],
        "cv_title_override": "Cloud Engineer",
        "keyword_boost": [
            "cloud infrastructure",
            "infrastructure as code",
            "cost optimisation",
            "high availability",
            "cloud migration",
        ],
    },
    RoleVariant.SOFTWARE_ENGINEER: {
        "variant": RoleVariant.SOFTWARE_ENGINEER.value,
        "top_skills": [
            "Python",
            "Java",
            "REST API",
            "Microservices",
            "SQL",
            "Git",
            "Docker",
            "React",
            "FastAPI",
            "Django",
        ],
        "section_order": ["experience", "projects", "skills", "education"],
        "cv_title_override": "Software Engineer",
        "keyword_boost": [
            "object-oriented design",
            "API development",
            "unit testing",
            "code review",
            "agile/scrum",
        ],
    },
    RoleVariant.GENERIC: {
        "variant": RoleVariant.GENERIC.value,
        "top_skills": [],
        "section_order": ["experience", "skills", "education"],
        "cv_title_override": "",
        "keyword_boost": [],
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_role(title: str, description: str = "") -> RoleVariant:
    """Return the best-matching RoleVariant for a job posting.

    Scores each variant by counting keyword hits in the combined
    lower-cased title + description text and returns the variant with
    the highest score.  Falls back to GENERIC when no keywords match.
    """
    combined = (f"{title} {description}").lower()

    scores: Dict[RoleVariant, int] = {v: 0 for v in RoleVariant if v != RoleVariant.GENERIC}

    for variant, keywords in _KEYWORDS.items():
        for kw in keywords:
            if kw in combined:
                scores[variant] += 1

    best_variant = max(scores, key=lambda v: scores[v])
    if scores[best_variant] == 0:
        return RoleVariant.GENERIC

    return best_variant


def get_variant_config(variant: RoleVariant) -> Dict:
    """Return a configuration dict for the given RoleVariant.

    Keys:
      - variant (str): canonical variant name
      - top_skills (list[str]): skills to emphasise first in the CV
      - section_order (list[str]): preferred section ordering
      - cv_title_override (str): headline to inject into the CV header
      - keyword_boost (list[str]): phrases to ensure prominent placement
    """
    return copy.deepcopy(_CONFIGS.get(variant, _CONFIGS[RoleVariant.GENERIC]))
