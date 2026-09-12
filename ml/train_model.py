import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import os

# Create output directories
os.makedirs('../models', exist_ok=True)

# Load dataset
print("Loading dataset...")
df = pd.read_csv('../data/ml_ready_dataset.csv')
print(f"Loaded {len(df)} samples")

# Features and targets
feature_cols = ['aoa_deg', 'sin_aoa', 'cos_aoa', 'log_Re', 'Ma', 'q_inf']
X = df[feature_cols].values
y_Cl = df['Cl'].values
y_Cd = df['Cd'].values

print(f"Features: {feature_cols}")
print(f"Cl range: [{y_Cl.min():.3f}, {y_Cl.max():.3f}]")
print(f"Cd range: [{y_Cd.min():.3f}, {y_Cd.max():.3f}]")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train models
print("\nTraining Cl model (Random Forest)...")
model_Cl = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
model_Cl.fit(X_scaled, y_Cl)

print("Training Cd model (Gradient Boosting)...")
model_Cd = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
model_Cd.fit(X_scaled, y_Cd)

# Leave-one-out CV
print("\nLeave-one-out cross-validation...")
loo = LeaveOneOut()
loo_scores_Cl = cross_val_score(model_Cl, X_scaled, y_Cl, cv=loo, scoring='r2')
loo_scores_Cd = cross_val_score(model_Cd, X_scaled, y_Cd, cv=loo, scoring='r2')

print(f"Cl LOO CV R²: {loo_scores_Cl.mean():.4f} ± {loo_scores_Cl.std():.4f}")
print(f"Cd LOO CV R²: {loo_scores_Cd.mean():.4f} ± {loo_scores_Cd.std():.4f}")

# Training performance
y_pred_Cl = model_Cl.predict(X_scaled)
y_pred_Cd = model_Cd.predict(X_scaled)

r2_Cl = r2_score(y_Cl, y_pred_Cl)
mae_Cl = mean_absolute_error(y_Cl, y_pred_Cl)
r2_Cd = r2_score(y_Cd, y_pred_Cd)
mae_Cd = mean_absolute_error(y_Cd, y_pred_Cd)

print(f"\nTraining Performance:")
print(f"Cl: R² = {r2_Cl:.4f}, MAE = {mae_Cl:.4f}")
print(f"Cd: R² = {r2_Cd:.4f}, MAE = {mae_Cd:.4f}")

# Save models
print("\nSaving models...")
joblib.dump(model_Cl, '../models/cl_surrogate.pkl')
joblib.dump(model_Cd, '../models/cd_surrogate.pkl')
joblib.dump(scaler, '../models/scaler.pkl')
print("Models saved to ../models/")

# Parity plots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(y_Cl, y_pred_Cl, s=100, edgecolors='k')
axes[0].plot([y_Cl.min(), y_Cl.max()], [y_Cl.min(), y_Cl.max()], 'r--')
axes[0].set_xlabel('Actual Cl')
axes[0].set_ylabel('Predicted Cl')
axes[0].set_title(f'Cl: R²={r2_Cl:.3f}, MAE={mae_Cl:.3f}')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_Cd, y_pred_Cd, s=100, edgecolors='k')
axes[1].plot([y_Cd.min(), y_Cd.max()], [y_Cd.min(), y_Cd.max()], 'r--')
axes[1].set_xlabel('Actual Cd')
axes[1].set_ylabel('Predicted Cd')
axes[1].set_title(f'Cd: R²={r2_Cd:.3f}, MAE={mae_Cd:.3f}')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/plots/parity_plots.png', dpi=150, bbox_inches='tight')
print("Parity plots saved to ../results/plots/parity_plots.png")

print("\n" + "="*50)
print("TRAINING COMPLETE")
print("="*50)
