"""
Multi-Domain Synthetic Dataset Generators for Adaptive Platform Testing.
Provides:
1. SaaS Subscription & Recurring Revenue (2024-2025)
2. Healthcare Clinical Admissions & Ward Operations (2025)
Both datasets include realistic relationships, domain-specific metrics, and auditable anomalies
(corrupt numeric records, invalid dates, and conflicting duplicates) for stress testing.
"""
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd


def generate_saas_dataset(base_dir: Path, seed: int = 42) -> Dict[str, Any]:
    """
    Generates SaaS Subscriptions & ARR dataset (2024-2025):
    - subscriptions.csv (fact table: subscription_id, account_id, start_date, end_date, plan_tier, mrr_amount, status)
    - accounts.csv (dimension: account_id, company_name, industry, country, seats)
    """
    random.seed(seed)
    np.random.seed(seed)
    out_dir = base_dir / "saas_subscriptions"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Accounts Dimension
    industries = ["Fintech", "Healthtech", "Edtech", "E-commerce", "Cybersecurity", "Logistics"]
    countries = ["United States", "United Kingdom", "Germany", "Canada", "Australia", "Brazil"]
    company_prefixes = ["Apex", "Cloud", "Data", "Nova", "Pulse", "Quantum", "Sphere", "Vertex", "Zenith", "Hyper"]
    company_suffixes = ["Systems", "Labs", "Tech", "Analytics", "Networks", "Cloud", "Solutions", "AI"]

    accounts = []
    account_ids = [f"ACC-{i:04d}" for i in range(1, 121)]
    for acc_id in account_ids:
        c_name = f"{random.choice(company_prefixes)} {random.choice(company_suffixes)}"
        accounts.append({
            "account_id": acc_id,
            "company_name": c_name,
            "industry": random.choice(industries),
            "country": random.choice(countries),
            "seats": random.choice([5, 10, 25, 50, 100, 250, 500])
        })

    # Add 1 conflicting duplicate in accounts to test quarantine
    accounts.append({
        "account_id": account_ids[5],
        "company_name": "Conflicting Account HQ",
        "industry": "Agriculture",
        "country": "Mars",
        "seats": 99999
    })

    df_accounts = pd.DataFrame(accounts)
    df_accounts.to_csv(out_dir / "accounts.csv", index=False)

    # 2. Subscriptions Fact
    plans = {
        "Starter": 49.0,
        "Growth": 199.0,
        "Professional": 499.0,
        "Enterprise": 1499.0
    }

    subscriptions = []
    sub_counter = 1
    start_base = datetime(2024, 1, 1)
    end_base = datetime(2025, 6, 30)

    for acc in account_ids:
        # Each account has 1 to 3 subscriptions over time
        num_subs = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        curr_date = start_base + timedelta(days=random.randint(0, 180))

        for _ in range(num_subs):
            if curr_date > end_base:
                break
            plan = random.choice(list(plans.keys()))
            mrr = plans[plan] * (1.0 + random.choice([0.0, -0.1, -0.2]))  # discount
            sub_id = f"SUB-{sub_counter:05d}"
            sub_counter += 1

            # Determine duration and status
            status_rand = random.random()
            if status_rand < 0.2:
                status = "CANCELLED"
                duration_days = random.randint(30, 120)
                sub_end = curr_date + timedelta(days=duration_days)
            elif status_rand < 0.35:
                status = "CHURNED"
                duration_days = random.randint(60, 240)
                sub_end = curr_date + timedelta(days=duration_days)
            else:
                status = "ACTIVE"
                sub_end = None

            subscriptions.append({
                "subscription_id": sub_id,
                "account_id": acc,
                "start_date": curr_date.strftime("%Y-%m-%d"),
                "end_date": sub_end.strftime("%Y-%m-%d") if sub_end else "",
                "plan_tier": plan,
                "mrr_amount": round(mrr, 2),
                "status": status
            })

            curr_date += timedelta(days=random.randint(60, 180))

    # Add controlled anomalies:
    # 2 corrupt numeric values (strings)
    subscriptions.append({
        "subscription_id": f"SUB-{sub_counter:05d}",
        "account_id": account_ids[0],
        "start_date": "2024-03-15",
        "end_date": "",
        "plan_tier": "Enterprise",
        "mrr_amount": "CORRUPTED_VALUE_N/A",
        "status": "ACTIVE"
    })
    sub_counter += 1
    # 2 invalid dates
    subscriptions.append({
        "subscription_id": f"SUB-{sub_counter:05d}",
        "account_id": account_ids[1],
        "start_date": "2024-99-99",
        "end_date": "",
        "plan_tier": "Starter",
        "mrr_amount": 49.0,
        "status": "ACTIVE"
    })

    df_subs = pd.DataFrame(subscriptions)
    df_subs.to_csv(out_dir / "subscriptions.csv", index=False)

    return {
        "dir": str(out_dir),
        "accounts_count": len(df_accounts),
        "subscriptions_count": len(df_subs)
    }


def generate_healthcare_dataset(base_dir: Path, seed: int = 123) -> Dict[str, Any]:
    """
    Generates Healthcare Admissions & Ward Capacity dataset (2025):
    - admissions.csv (fact table: encounter_id, patient_id, ward_id, admit_date, discharge_date, length_of_stay, total_cost, admission_type, status)
    - wards.csv (dimension: ward_id, ward_name, department, bed_count)
    - physicians.csv (dimension: physician_id, physician_name, specialty, primary_ward_id)
    """
    random.seed(seed)
    np.random.seed(seed)
    out_dir = base_dir / "healthcare_admissions"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Wards Dimension
    wards = [
        {"ward_id": "WRD-01", "ward_name": "Cardiology Inpatient", "department": "Cardiovascular", "bed_count": 40},
        {"ward_id": "WRD-02", "ward_name": "Intensive Care Unit (ICU)", "department": "Critical Care", "bed_count": 20},
        {"ward_id": "WRD-03", "ward_name": "General Surgery Ward", "department": "Surgery", "bed_count": 50},
        {"ward_id": "WRD-04", "ward_name": "Oncology & Hematology", "department": "Internal Medicine", "bed_count": 35},
        {"ward_id": "WRD-05", "ward_name": "Pediatrics Unit", "department": "Pediatrics", "bed_count": 30},
        {"ward_id": "WRD-06", "ward_name": "Neurology & Stroke", "department": "Neurosciences", "bed_count": 25},
    ]
    df_wards = pd.DataFrame(wards)
    df_wards.to_csv(out_dir / "wards.csv", index=False)

    # 2. Physicians Dimension
    physicians = [
        {"physician_id": "DR-101", "physician_name": "Dr. Sarah Adams", "specialty": "Cardiology", "ward_id": "WRD-01"},
        {"physician_id": "DR-102", "physician_name": "Dr. Marcus Vance", "specialty": "Critical Care", "ward_id": "WRD-02"},
        {"physician_id": "DR-103", "physician_name": "Dr. Elena Rostova", "specialty": "General Surgery", "ward_id": "WRD-03"},
        {"physician_id": "DR-104", "physician_name": "Dr. Tariq Al-Mansoor", "specialty": "Oncology", "ward_id": "WRD-04"},
        {"physician_id": "DR-105", "physician_name": "Dr. Chloe Dubois", "specialty": "Pediatrics", "ward_id": "WRD-05"},
        {"physician_id": "DR-106", "physician_name": "Dr. Kenji Sato", "specialty": "Neurology", "ward_id": "WRD-06"},
    ]
    df_physicians = pd.DataFrame(physicians)
    df_physicians.to_csv(out_dir / "physicians.csv", index=False)

    # 3. Admissions Fact Table (Jan to May 2025)
    admission_types = ["EMERGENCY", "ELECTIVE", "URGENT", "NEWBORN"]
    cost_ranges = {
        "WRD-01": (3500, 12000),
        "WRD-02": (8000, 28000),
        "WRD-03": (4000, 15000),
        "WRD-04": (5000, 20000),
        "WRD-05": (2000, 8000),
        "WRD-06": (4500, 16000),
    }

    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 5, 31)
    total_days = (end_date - start_date).days

    admissions = []
    patient_ids = [f"PAT-{i:05d}" for i in range(1, 351)]

    encounter_id = 10001
    for day_offset in range(total_days):
        current_day = start_date + timedelta(days=day_offset)
        daily_count = random.randint(4, 12)
        for _ in range(daily_count):
            patient = random.choice(patient_ids)
            ward = random.choice(wards)["ward_id"]
            los = random.randint(1, 14)
            disch_date = current_day + timedelta(days=los)
            min_c, max_c = cost_ranges[ward]
            cost = round(random.uniform(min_c, max_c) + (los * 400.0), 2)
            adm_type = random.choice(admission_types)

            admissions.append({
                "encounter_id": f"ENC-{encounter_id}",
                "patient_id": patient,
                "ward_id": ward,
                "admit_date": current_day.strftime("%Y-%m-%d"),
                "discharge_date": disch_date.strftime("%Y-%m-%d"),
                "length_of_stay": los,
                "total_cost": cost,
                "admission_type": adm_type,
                "status": "DISCHARGED" if disch_date <= end_date else "ACTIVE"
            })
            encounter_id += 1

    # Add anomalies:
    admissions.append({
        "encounter_id": f"ENC-{encounter_id}",
        "patient_id": patient_ids[0],
        "ward_id": "WRD-01",
        "admit_date": "2025-03-01",
        "discharge_date": "2025-03-05",
        "length_of_stay": 4,
        "total_cost": "UNKNOWN_BILLING_ERR",
        "admission_type": "EMERGENCY",
        "status": "DISCHARGED"
    })
    encounter_id += 1

    admissions.append({
        "encounter_id": f"ENC-{encounter_id}",
        "patient_id": patient_ids[1],
        "ward_id": "WRD-02",
        "admit_date": "2025-02-31",  # Invalid day for Feb
        "discharge_date": "2025-03-05",
        "length_of_stay": 4,
        "total_cost": 9500.0,
        "admission_type": "URGENT",
        "status": "DISCHARGED"
    })

    df_admissions = pd.DataFrame(admissions)
    df_admissions.to_csv(out_dir / "admissions.csv", index=False)

    return {
        "dir": str(out_dir),
        "wards_count": len(df_wards),
        "physicians_count": len(df_physicians),
        "admissions_count": len(df_admissions)
    }
