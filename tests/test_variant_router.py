"""Tests for app.services.cv.variant_router."""

import pytest

from app.services.cv.variant_router import RoleVariant, classify_role, get_variant_config


def test_devops_job_classified_as_devops():
    variant = classify_role(
        "DevOps Engineer",
        "We need someone with Kubernetes Docker CI/CD experience.",
    )
    assert variant == RoleVariant.DEVOPS


def test_mlops_job_classified():
    variant = classify_role(
        "MLOps Engineer",
        "Responsibilities include MLflow model deployment and monitoring.",
    )
    assert variant == RoleVariant.MLOPS


def test_it_support_classified():
    variant = classify_role("IT Support Specialist")
    assert variant == RoleVariant.IT_SUPPORT


def test_generic_fallback():
    variant = classify_role("Manager")
    assert variant == RoleVariant.GENERIC


def test_variant_config_has_required_keys():
    cfg = get_variant_config(RoleVariant.DEVOPS)
    assert "top_skills" in cfg
    assert "cv_title_override" in cfg
    assert "keyword_boost" in cfg
    assert "section_order" in cfg
    assert "variant" in cfg


def test_classify_uses_description():
    variant = classify_role(
        "Engineer",
        "kubernetes terraform ci/cd pipeline management",
    )
    assert variant == RoleVariant.DEVOPS


def test_all_variants_have_configs():
    for variant in RoleVariant:
        cfg = get_variant_config(variant)
        assert isinstance(cfg.get("top_skills"), list)
        assert isinstance(cfg.get("keyword_boost"), list)
        assert isinstance(cfg.get("section_order"), list)


def test_variant_config_returns_copy():
    cfg1 = get_variant_config(RoleVariant.DEVOPS)
    cfg2 = get_variant_config(RoleVariant.DEVOPS)
    cfg1["top_skills"].append("INJECTED")
    assert "INJECTED" not in cfg2["top_skills"]


def test_it_admin_classified():
    variant = classify_role(
        "System Administrator",
        "Manage Active Directory, Group Policy, and Windows Server 2022.",
    )
    assert variant == RoleVariant.IT_ADMIN


def test_data_engineer_classified():
    variant = classify_role(
        "Data Engineer",
        "Build ETL pipelines using Apache Spark, Kafka, and Airflow.",
    )
    assert variant == RoleVariant.DATA_ENGINEER
