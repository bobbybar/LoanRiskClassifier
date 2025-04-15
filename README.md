# LoanRiskClassifier

# LendingClub 1-Year Synthetic Dataset

This dataset is a synthetic version of a one-year snapshot of LendingClub loan data, intended for educational and exploratory data science projects.

## 📄 Column Descriptions

| Column Name        | Description |
|--------------------|-------------|
| **loan_amnt**      | The amount of money the borrower requested. |
| **term**           | The length of the loan: typically **36 months** or **60 months**. |
| **int_rate**       | The interest rate on the loan — higher rates usually mean higher risk. |
| **grade**          | LendingClub’s internal credit grade (A-G), where **A is best** and **G is riskiest**. |
| **emp_length**     | How long the borrower has been employed (e.g., "< 1 year", "10+ years"). |
| **home_ownership** | Whether the borrower **rents**, **owns**, or has a **mortgage**. |
| **annual_inc**     | Borrower's reported annual income. |
| **purpose**        | The purpose of the loan (e.g., **credit_card**, **debt_consolidation**, **home_improvement**). |
| **addr_state**     | The U.S. state where the borrower lives. |
| **dti**            | **Debt-to-Income ratio** — how much debt someone has relative to income. Higher DTI = riskier. |
| **delinq_2yrs**    | Number of times borrower was **delinquent** (late on payment) in the last 2 years. |
| **revol_util**     | Utilization rate of revolving credit (like credit cards) — expressed as a percentage. |
| **loan_status**    | Loan outcome: **Fully Paid** or **Charged Off**. |
| **default**        | Binary target variable: `1` = **Default**, `0` = **Fully Paid**. (Used for modeling) ✅

---

## 🔍 Use Cases

- Exploratory Data Analysis (EDA)
- Loan default risk modeling
- Classification practice (Logistic Regression, Random Forest, XGBoost, etc.)
- Dashboard creation with Streamlit