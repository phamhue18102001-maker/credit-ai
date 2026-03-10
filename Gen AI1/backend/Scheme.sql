CREATE TABLE customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    age INTEGER,
    annual_income REAL,
    monthly_salary REAL,
    num_bank_accounts INTEGER,
    num_credit_card INTEGER,
    interest_rate REAL,
    num_of_loan INTEGER,
    delay_from_due_date INTEGER,
    outstanding_debt REAL,
    credit_utilization_ratio REAL,
    credit_history_age INTEGER,
    total_emi_per_month REAL,
    amount_invested_monthly REAL,
    monthly_balance REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Bảng lưu kết quả dự đoán từ AI model
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER,
    credit_score TEXT,
    probability REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);