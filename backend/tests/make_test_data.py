import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 300

# ---------- File 1: date + ID columns (classification) ----------
age = rng.integers(19, 70, n)
visits = rng.integers(1, 30, n)
day = rng.integers(1, 32, n)  # plain numbers 1-31, must NOT be treated as a date
price = np.round(rng.uniform(10, 500, n), 2)  # unique continuous, must NOT be treated as an ID
city = rng.choice(["Nashik", "Pune", "Mumbai", "Nagpur"], n)

dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 700, n), unit="D")

score = -0.05 * age + 0.08 * visits + rng.normal(0, 1, n)
churn = np.where(score < np.percentile(score, 40), "Yes", "No")

df1 = pd.DataFrame({
    "row_id": np.arange(1, n + 1),                              # plain 1..N counter
    "account_no": rng.permutation(np.arange(100000, 100000 + n)),  # numeric ID
    "customer_id": [f"C{i:04d}" for i in range(1, n + 1)],      # text ID
    "order_date": dates.strftime("%Y-%m-%d"),                   # real date column
    "age": age,
    "visits": visits,
    "day": day,
    "price": price,
    "city": city,
    "churn": churn,
})
df1.to_csv("test_dates_ids.csv", index=False)

# ---------- File 2: regression with negative target ----------
units = rng.integers(1, 50, n)
unit_price = np.round(rng.uniform(10, 100, n), 2)
discount = np.round(rng.uniform(0, 0.4, n), 2)
region = rng.choice(["East", "West", "North", "South"], n)

profit = units * unit_price * (0.3 - discount) - 150 + rng.normal(0, 20, n)

df2 = pd.DataFrame({
    "units": units,
    "unit_price": unit_price,
    "discount": discount,
    "region": region,
    "profit": np.round(profit, 2),
})
df2.to_csv("test_negative_target.csv", index=False)

print("profit min:", df2["profit"].min())  # should be negative
print("Done: test_dates_ids.csv, test_negative_target.csv")