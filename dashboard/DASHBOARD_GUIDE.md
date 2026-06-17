# Power BI Dashboard Guide

## Data Source

Import `data/processed/powerbi_export.csv` into Power BI Desktop.

**File:** `powerbi_export.csv` — 7,032 rows, 27 columns

## Dashboard Pages

### Page 1: Executive Summary

| Visual | Type | Fields |
|--------|------|--------|
| Churn Rate | Card | `Churn` (count: Yes / total) |
| Total Customers | Card | Count of `CustomerID` |
| Monthly Revenue at Risk | Card | SUM of `MonthlyCharges` filtered to `Churn = Yes` |
| Avg Monthly Charges | Card | AVERAGE of `MonthlyCharges` |
| Churn by Segment | Stacked Bar | Axis: `Segment`, Value: Count of `CustomerID`, Legend: `Churn` |
| Revenue at Risk by Segment | Bar | Axis: `Segment`, Value: SUM of `MonthlyCharges` (filter: Churn=Yes) |
| Churn Rate by Tenure Group | Column | Axis: `TenureGroup`, Value: % of `Churn = Yes` |

### Page 2: Churn Drivers

| Visual | Type | Fields |
|--------|------|--------|
| Contract Type Impact | Donut | Legend: `Contract`, Values: Count, Details: % Churn |
| Payment Method Impact | Horizontal Bar | Axis: `PaymentMethod`, Value: Count, Color: `Churn` |
| Service Impact Matrix | Matrix | Rows: `InternetService`, Columns: `OnlineSecurity`, Value: Churn Rate % |
| Monthly Charges Distribution | Histogram | `MonthlyCharges` by Churn | 

### Page 3: Customer Risk

| Visual | Type | Fields |
|--------|------|--------|
| Risk Score Distribution | Histogram | `RiskScore` by 0.1 buckets, Color: `Churn` |
| Risk Band Breakdown | Donut | Legend: `RiskBand`, Values: Count |
| High-Value At-Risk Table | Table | `CustomerID`, `tenure`, `MonthlyCharges`, `Contract`, `RiskScore`, `ChurnProbability` |
| Retention Opportunity Matrix | Scatter | X: `RiskScore`, Y: `MonthlyCharges`, Size: `tenure`, Color: `Segment` |
| Slicers | Slicer | `Segment`, `RiskBand`, `Contract` |

### Page 4: Retention Opportunities

| Visual | Type | Fields |
|--------|------|--------|
| Retention Priority | Bar | Axis: `RetentionPriority`, Value: Count, Color: `Churn` |
| Top Retention Targets | Table | Top 50 high-value at-risk customers |
| Action by Segment | Matrix | Rows: `Segment`, Columns: `RetentionPriority`, Value: Count |
| Revenue Impact | Calculated | DAX measures for projected savings |

## DAX Measures to Create

```dax
Churn Rate = DIVIDE(
    COUNTROWS(FILTER('powerbi_export', 'powerbi_export'[Churn] = "Yes")),
    COUNTROWS('powerbi_export')
)

Revenue at Risk = CALCULATE(
    SUM('powerbi_export'[MonthlyCharges]),
    'powerbi_export'[Churn] = "Yes"
)

High Value At Risk = COUNTROWS(
    FILTER(
        'powerbi_export',
        'powerbi_export'[RiskScore] >= 0.6 &&
        'powerbi_export'[MonthlyCharges] >= 70
    )
)

Retention Opportunity = [Revenue at Risk] * 0.3
// Assumes 30% of at-risk revenue is retainable
```

## Building the Dashboard

1. **Get Data** → CSV → Select `powerbi_export.csv`
2. **Data Types:** Ensure `MonthlyCharges`, `TotalCharges`, `tenure`, `RiskScore` are Decimal/Whole Number
3. **Create Measures** using the DAX above
4. **Design pages** as specified
5. **Apply theme:** Corporate blue/white or use the "Executive" built-in theme

## Export for Portfolio

- Save as `.pbix`: `Telco_Churn_Dashboard.pbix`
- Export key pages as PDF/PNG for portfolio
- Export as PowerPoint for presentations
