"""
Deterministic synthetic data generator for sales, customers, products, and campaigns.
Injects intentional real-world data quality issues and known ground truth for validation.
"""
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl import Workbook


def generate_demo_dataset(target_dir: Path, seed: int = 42) -> dict:
    """
    Generate synthetic data files with known ground truth and realistic anomalies:
    - orders.csv
    - customers.csv
    - products.xlsx (sheets: 'Products', 'Categories')
    - campaigns.csv
    - ground_truth.json
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    # 1. Campaigns
    campaigns_data = [
        {"campaign_id": "CMP-01", "campaign_name": "Summer Launch", "start_date": "2026-01-05", "end_date": "2026-02-15", "budget": 15000.0},
        {"campaign_id": "CMP-02", "campaign_name": "Back to School", "start_date": "2026-02-20", "end_date": "2026-03-25", "budget": 22000.0},
        {"campaign_id": "CMP-03", "campaign_name": "B2B Industrial Renewal", "start_date": "2026-03-01", "end_date": "2026-04-10", "budget": 30000.0},
        {"campaign_id": "CMP-04", "campaign_name": "Autumn Cyber", "start_date": "2026-05-01", "end_date": "2026-05-20", "budget": 18000.0},
    ]
    df_campaigns = pd.DataFrame(campaigns_data)
    campaigns_path = target_dir / "campaigns.csv"
    df_campaigns.to_csv(campaigns_path, index=False)

    # 2. Categories & Products (Excel with 2 sheets)
    categories_data = [
        {"category_id": "CAT_TECH", "category_name": "Technology & Equipment", "department": "Hardware"},
        {"category_id": "CAT_OFFICE", "category_name": "Office Supplies", "department": "Consumibles"},
        {"category_id": "CAT_FURN", "category_name": "Ergonomic Furniture", "department": "Infraestructura"},
        {"category_id": "CAT_SERV", "category_name": "Services & Licenses", "department": "Software"},
    ]
    df_categories = pd.DataFrame(categories_data)

    products_data = [
        {"product_id": "PROD_001", "product_name": "Enterprise Rack Server", "category_id": "CAT_TECH", "cost_price": 3200.0, "base_price": 4500.0},
        {"product_id": "PROD_002", "product_name": "Professional Workstation", "category_id": "CAT_TECH", "cost_price": 1400.0, "base_price": 2100.0},
        {"product_id": "PROD_003", "product_name": "Ultrawide 4K Monitor", "category_id": "CAT_TECH", "cost_price": 450.0, "base_price": 750.0},
        {"product_id": "PROD_004", "product_name": "Ergonomic Pro Chair", "category_id": "CAT_FURN", "cost_price": 210.0, "base_price": 380.0},
        {"product_id": "PROD_005", "product_name": "Motorized Standing Desk", "category_id": "CAT_FURN", "cost_price": 390.0, "base_price": 650.0},
        {"product_id": "PROD_006", "product_name": "Corporate Stationery Pack", "category_id": "CAT_OFFICE", "cost_price": 25.0, "base_price": 55.0},
        {"product_id": "PROD_007", "product_name": "High-Capacity Laser Toner", "category_id": "CAT_OFFICE", "cost_price": 60.0, "base_price": 120.0},
        {"product_id": "PROD_008", "product_name": "Annual Cloud Subscription", "category_id": "CAT_SERV", "cost_price": 800.0, "base_price": 1400.0},
    ]
    df_products = pd.DataFrame(products_data)

    products_path = target_dir / "products.xlsx"
    with pd.ExcelWriter(products_path, engine="openpyxl") as writer:
        df_products.to_excel(writer, sheet_name="Products", index=False)
        df_categories.to_excel(writer, sheet_name="Categories", index=False)

    # 3. Customers
    # Preserve leading zeros! e.g. "00101"
    regions = ["Metropolitan", "North", "Central", "South"]
    segments = ["Corporate B2B", "Direct Retail", "Institutional"]
    customers_list = []
    num_customers = 80

    for i in range(1, num_customers + 1):
        cust_id = f"{i:05d}"  # 00001, 00002...
        customers_list.append({
            "customer_id": cust_id,
            "customer_name": f"Corporate Client {i}",
            "segment": np.random.choice(segments, p=[0.4, 0.45, 0.15]),
            "region": np.random.choice(regions),
            "signup_date": (datetime(2025, 6, 1) + timedelta(days=int(np.random.randint(0, 240)))).strftime("%Y-%m-%d")
        })

    # INTENTIONAL ISSUE: Duplicate customer ID with differing address
    duplicate_cust = customers_list[14].copy()
    duplicate_cust["customer_name"] = duplicate_cust["customer_name"] + " (Secondary Branch)"
    duplicate_cust["region"] = "South"
    customers_list.append(duplicate_cust)

    df_customers = pd.DataFrame(customers_list)
    customers_path = target_dir / "customers.csv"
    df_customers.to_csv(customers_path, index=False)

    # 4. Orders
    # Dates from 2026-01-01 to 2026-06-04
    # Notice: June is incomplete (only 4 days)
    # Notice: In April and May 2026, wholesale (Canal Mayorista / B2B) drops dramatically!
    orders_list = []
    order_counter = 10000

    channels = ["Wholesale / B2B", "Retail / Stores", "Online / Direct"]
    product_map = {p["product_id"]: p for p in products_data}
    valid_cust_ids = [c["customer_id"] for c in customers_list[:num_customers]]

    # Simulation timeline: Jan 1 to June 4, 2026
    start_date = datetime(2026, 1, 1)
    end_date = datetime(2026, 6, 4)
    curr = start_date

    # Track customer first purchase for new vs recurring logic
    customer_first_purchase = {}

    while curr <= end_date:
        year_month = curr.strftime("%Y-%m")
        # Incomplete period in June: only 4 days
        is_june = (curr.month == 6)

        # Baseline: Jan, Feb, Mar (months 1, 2, 3)
        # Drop period: Apr, May (months 4, 5)
        # In months 4 and 5, Mayorista orders drop from 18/day to 4/day!
        num_orders_today = np.random.randint(15, 25)
        if is_june:
            num_orders_today = np.random.randint(10, 18)

        for _ in range(num_orders_today):
            order_counter += 1
            order_id = f"ORD-{order_counter}"
            order_date_str = curr.strftime("%Y-%m-%d")

            channel = np.random.choice(channels, p=[0.45, 0.35, 0.20])
            # If in April or May and channel is Mayorista, suppress orders significantly
            if curr.month in [4, 5] and channel == "Wholesale / B2B":
                if np.random.rand() > 0.25:  # 75% dropped in Mayorista!
                    continue

            # Pick customer
            # Some customers repeat
            cust_id = np.random.choice(valid_cust_ids)
            if cust_id not in customer_first_purchase:
                customer_first_purchase[cust_id] = order_date_str

            # Pick product
            prod_id = np.random.choice(list(product_map.keys()))
            prod = product_map[prod_id]

            # Units
            if channel == "Wholesale / B2B":
                units = int(np.random.randint(5, 25))
            else:
                units = int(np.random.randint(1, 4))

            unit_price = prod["base_price"]
            gross_sales = float(units * unit_price)

            # Discount
            discount_pct = np.random.choice([0.0, 0.05, 0.10, 0.15], p=[0.5, 0.25, 0.15, 0.10])
            discount_amount = round(gross_sales * discount_pct, 2)
            net_sales = round(gross_sales - discount_amount, 2)

            # Order Status: Mostly completed, ~4% cancelled, ~2% refunded
            status = np.random.choice(["COMPLETED", "CANCELLED", "REFUNDED"], p=[0.94, 0.04, 0.02])

            # Campaign attribution
            campaign_id = None
            if curr.month == 1:
                campaign_id = "CMP-01" if np.random.rand() < 0.6 else None
            elif curr.month == 2:
                campaign_id = "CMP-02" if np.random.rand() < 0.5 else None
            elif curr.month == 3:
                campaign_id = "CMP-03" if np.random.rand() < 0.4 else None
            elif curr.month == 5:
                campaign_id = "CMP-04" if np.random.rand() < 0.5 else None

            orders_list.append({
                "order_id": order_id,
                "order_date": order_date_str,
                "customer_id": cust_id,
                "product_id": prod_id,
                "campaign_id": campaign_id,
                "channel": channel,
                "units": units,
                "unit_price": unit_price,
                "discount_amount": discount_amount if np.random.rand() > 0.05 else None,  # intentional missing
                "gross_sales": gross_sales,
                "net_sales": net_sales,
                "status": status
            })

        curr += timedelta(days=1)

    # INTENTIONAL ISSUE: Orphan records (orders with customer_id or product_id that don't exist)
    for k in range(5):
        order_counter += 1
        orders_list.append({
            "order_id": f"ORD-{order_counter}",
            "order_date": "2026-03-15",
            "customer_id": "00999",  # Orphan customer!
            "product_id": "PROD_001",
            "campaign_id": None,
            "channel": "Retail / Stores",
            "units": 2,
            "unit_price": 4500.0,
            "discount_amount": 0.0,
            "gross_sales": 9000.0,
            "net_sales": 9000.0,
            "status": "COMPLETED"
        })

    for k in range(3):
        order_counter += 1
        orders_list.append({
            "order_id": f"ORD-{order_counter}",
            "order_date": "2026-04-12",
            "customer_id": "00005",
            "product_id": "PROD_999",  # Orphan product!
            "campaign_id": None,
            "channel": "Online / Direct",
            "units": 1,
            "unit_price": 1000.0,
            "discount_amount": 0.0,
            "gross_sales": 1000.0,
            "net_sales": 1000.0,
            "status": "COMPLETED"
        })

    # INTENTIONAL ISSUE: Invalid dates
    order_counter += 1
    orders_list.append({
        "order_id": f"ORD-{order_counter}",
        "order_date": "2026-02-30",  # Invalid date!
        "customer_id": "00001",
        "product_id": "PROD_002",
        "campaign_id": None,
        "channel": "Retail / Stores",
        "units": 1,
        "unit_price": 2100.0,
        "discount_amount": 0.0,
        "gross_sales": 2100.0,
        "net_sales": 2100.0,
        "status": "COMPLETED"
    })

    order_counter += 1
    orders_list.append({
        "order_id": f"ORD-{order_counter}",
        "order_date": "CORRUPT_DATE",  # Unparseable date!
        "customer_id": "00002",
        "product_id": "PROD_003",
        "campaign_id": None,
        "channel": "Online / Direct",
        "units": 1,
        "unit_price": 750.0,
        "discount_amount": 0.0,
        "gross_sales": 750.0,
        "net_sales": 750.0,
        "status": "COMPLETED"
    })

    df_orders = pd.DataFrame(orders_list)
    orders_path = target_dir / "orders.csv"
    df_orders.to_csv(orders_path, index=False)

    # 5. Compute Exact Ground Truth for Valid Orders (Status == COMPLETED, valid date, non-orphan)
    # Parse dates safely for ground truth calculation
    valid_mask = (
        (df_orders["status"] == "COMPLETED") &
        (df_orders["order_date"].str.match(r"^\d{4}-\d{2}-\d{2}$")) &
        (~df_orders["customer_id"].isin(["00999"])) &
        (~df_orders["product_id"].isin(["PROD_999"]))
    )
    df_valid = df_orders[valid_mask].copy()
    # Filter out Feb 30 if string matched regex
    df_valid = df_valid[pd.to_datetime(df_valid["order_date"], errors="coerce").notna()].copy()
    df_valid["order_date_dt"] = pd.to_datetime(df_valid["order_date"])
    df_valid["year_month"] = df_valid["order_date_dt"].dt.strftime("%Y-%m")

    # Baseline Period: Q1 2026 (Jan - Mar) vs Comparison Period: April - May 2026
    # Monthly net sales
    monthly_sales = df_valid.groupby("year_month")["net_sales"].sum().to_dict()
    channel_monthly_sales = df_valid.groupby(["year_month", "channel"])["net_sales"].sum().unstack(fill_value=0).to_dict()

    # Pre-drop baseline (March 2026) vs Post-drop (May 2026) or Q1 avg vs April-May avg
    sales_mar = monthly_sales.get("2026-03", 0.0)
    sales_may = monthly_sales.get("2026-05", 0.0)
    diff_total = sales_may - sales_mar

    # Absolute contribution by channel from March to May
    channel_mar = df_valid[df_valid["year_month"] == "2026-03"].groupby("channel")["net_sales"].sum()
    channel_may = df_valid[df_valid["year_month"] == "2026-05"].groupby("channel")["net_sales"].sum()
    channel_diff = (channel_may - channel_mar).to_dict()

    ground_truth = {
        "dataset_seed": seed,
        "total_orders_generated": len(df_orders),
        "total_customers": len(df_customers),
        "total_products": len(df_products),
        "orphan_orders_count": 8,
        "invalid_date_orders_count": 2,
        "duplicate_customer_ids": ["00142"],
        "incomplete_period": {
            "month": "2026-06",
            "days_present": 4,
            "is_incomplete": True
        },
        "monthly_net_sales": monthly_sales,
        "channel_monthly_net_sales": {k: {m: v for m, v in vals.items()} for k, vals in channel_monthly_sales.items()},
        "march_net_sales": float(sales_mar),
        "may_net_sales": float(sales_may),
        "drop_amount_mar_to_may": float(diff_total),
        "channel_change_mar_to_may": {k: float(v) for k, v in channel_diff.items()},
        "primary_driver_channel": "Wholesale / B2B"
    }

    gt_path = target_dir / "ground_truth.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    return ground_truth


if __name__ == "__main__":
    demo_dir = Path(r"C:\Users\sebab\.gemini\antigravity\scratch\adaptive_analytics_poc\data\demo")
    gt = generate_demo_dataset(demo_dir)
    print(f"Generated demo dataset with {gt['total_orders_generated']} orders.")
    print(f"March Sales: {gt['march_net_sales']:,.2f}")
    print(f"May Sales: {gt['may_net_sales']:,.2f}")
    print(f"Drop March->May: {gt['drop_amount_mar_to_may']:,.2f}")
    print(f"Channel changes: {gt['channel_change_mar_to_may']}")
