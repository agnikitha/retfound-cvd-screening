import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

d_tr = np.load('D:/retfound anti/cimt_train_embeds.npz')
d_te = np.load('D:/retfound anti/cimt_test_embeds.npz')

# 1. Age-only
reg_age = Ridge(alpha=10.0, random_state=42).fit(d_tr['age'].reshape(-1, 1), d_tr['thickness'])
pred_age = reg_age.predict(d_te['age'].reshape(-1, 1))
mae_age = mean_absolute_error(d_te['thickness'], pred_age)
r_age, _ = pearsonr(d_te['thickness'], pred_age)

# 2. Age + Gender
X_tr_demo = np.column_stack([d_tr['age'], d_tr['gender']])
X_te_demo = np.column_stack([d_te['age'], d_te['gender']])
reg_demo = Ridge(alpha=10.0, random_state=42).fit(X_tr_demo, d_tr['thickness'])
pred_demo = reg_demo.predict(X_te_demo)
mae_demo = mean_absolute_error(d_te['thickness'], pred_demo)
r_demo, _ = pearsonr(d_te['thickness'], pred_demo)

# 3. Retina-only (Bilateral)
X_tr_ret = np.concatenate([d_tr['left_embeds'], d_tr['right_embeds']], axis=1)
X_te_ret = np.concatenate([d_te['left_embeds'], d_te['right_embeds']], axis=1)
reg_ret = Ridge(alpha=10.0, random_state=42).fit(X_tr_ret, d_tr['thickness'])
pred_ret = reg_ret.predict(X_te_ret)
mae_ret = mean_absolute_error(d_te['thickness'], pred_ret)
r_ret, _ = pearsonr(d_te['thickness'], pred_ret)

print(f"Age-Only:     MAE = {mae_age:.4f} mm | Pearson r = {r_age:.4f}")
print(f"Age+Gender:   MAE = {mae_demo:.4f} mm | Pearson r = {r_demo:.4f}")
print(f"Retina-Only:  MAE = {mae_ret:.4f} mm | Pearson r = {r_ret:.4f}")
