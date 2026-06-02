# Credit Scoring Business Understanding
# Basel II and Model Interpretability

Basel II requires financial institutions to maintain transparent, well-documented, and auditable risk assessment processes. Credit scoring models directly influence lending decisions and regulatory capital requirements. Therefore, models must be explainable so that banks can justify why a customer was approved or rejected for credit.

Interpretability also supports regulatory compliance, model validation, risk management, and customer protection. Simpler models such as Logistic Regression are often preferred because their decisions can be explained clearly to regulators and stakeholders.

# Why a Proxy Target Variable is Necessary

The dataset does not contain a direct default indicator showing whether a customer repaid a loan or failed to repay it. Since supervised learning requires labeled examples, a proxy variable must be created.

In this project, customer behavior will be analyzed using Recency, Frequency, and Monetary (RFM) metrics. Customers exhibiting low engagement and low transaction activity may be considered higher risk and assigned a proxy default label.

However, proxy variables introduce business risks because they may not perfectly represent actual credit default behavior. Incorrect labeling can lead to inaccurate risk predictions and potentially unfair lending decisions.

# Trade-Off Between Interpretable and High-Performance Models

Interpretable models such as Logistic Regression provide transparency and are easier to explain to regulators. Their coefficients directly show how features influence credit risk.

More advanced models such as Random Forests and Gradient Boosting typically achieve higher predictive performance but operate as black-box systems. Although these models may improve risk prediction accuracy, they require additional explainability techniques such as SHAP values to satisfy regulatory expectations.

Financial institutions must balance predictive power with regulatory compliance, fairness, and transparency when selecting a final credit scoring model.