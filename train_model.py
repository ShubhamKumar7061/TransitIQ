"""
TransitIQ — Model Training Script
Run this once: python train_model.py
Generates: dataset.csv, model_weights.json, metrics.json
Pure Python + math + Model 
"""
import csv, json, random, math, os
from datetime import datetime, timedelta

random.seed(42)

# ── 1. Generate Realistic Dataset ─────────────────────────────────────────
print("📦 Generating dataset...")

MODES    = ['Bus', 'Train', 'Airline']
ROUTES   = ['Fastest', 'Easiest']
PAYMENTS = ['Cash', 'UPI', 'Card']
CLASSES  = ['Economy', 'Business', 'First Class']

def true_demand(price, ttime, day, holiday, hour, mode, route, payment, tclass):
    """Ground truth formula (hidden from model — model must learn this)"""
    d = 220
    d -= price * 0.014
    d -= ttime * 0.09
    d += day * 40
    d += holiday * 70
    if 7 <= hour <= 9:   d += 55
    if 17 <= hour <= 20: d += 65
    if hour <= 4:        d -= 50
    d += {'Bus':8,'Train':28,'Airline':12}[mode]
    d += {'Economy':18,'Business':-8,'First Class':-25}[tclass]
    d += {'Fastest':12,'Easiest':5}[route]
    d += {'UPI':10,'Card':6,'Cash':0}[payment]
    d += random.gauss(0, 18)   # realistic noise
    return max(5, min(500, int(d)))

rows = []
for _ in range(3000):
    price   = random.randint(100, 9500)
    ttime   = random.randint(30, 700)
    day     = random.randint(0, 1)
    holiday = random.randint(0, 1)
    hour    = random.randint(0, 23)
    mode    = random.choice(MODES)
    route   = random.choice(ROUTES)
    pay     = random.choice(PAYMENTS)
    tclass  = random.choice(CLASSES)

    mode_enc  = MODES.index(mode)
    route_enc = ROUTES.index(route)
    pay_enc   = PAYMENTS.index(pay)
    class_enc = CLASSES.index(tclass)

    demand = true_demand(price, ttime, day, holiday, hour, mode, route, pay, tclass)
    rows.append([price, ttime, day, holiday, hour,
                 mode_enc, route_enc, pay_enc, class_enc, demand,
                 mode, route, pay, tclass])

header = ['ticket_price','travel_time','day_type','is_holiday','dep_hour',
          'mode_enc','route_enc','pay_enc','class_enc','demand',
          'mode','route_type','payment','travel_class']

with open('dataset.csv','w',newline='') as f:
    w = csv.writer(f); w.writerow(header); w.writerows(rows)

print(f"  ✅ dataset.csv saved — {len(rows)} records")

# ── 2. Train Decision Tree Regressor (pure Python) ────────────────────────
print("🌲 Training Decision Tree Regressor...")

FEATURES = ['ticket_price','travel_time','day_type','is_holiday','dep_hour',
            'mode_enc','route_enc','pay_enc','class_enc']

# Load data
X, y = [], []
with open('dataset.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        X.append([float(row[k]) for k in FEATURES])
        y.append(float(row['demand']))

# Train/test split (80/20)
n = len(X)
split = int(n * 0.8)
indices = list(range(n)); random.shuffle(indices)
train_idx = indices[:split]; test_idx = indices[split:]
X_train = [X[i] for i in train_idx]; y_train = [y[i] for i in train_idx]
X_test  = [X[i] for i in test_idx];  y_test  = [y[i] for i in test_idx]

# ── Pure Python Decision Tree ──────────────────────────────────────────────
class Node:
    def __init__(self):
        self.feature = None; self.threshold = None
        self.left = None; self.right = None; self.value = None

def mse(vals):
    if not vals: return 0
    m = sum(vals)/len(vals)
    return sum((v-m)**2 for v in vals)/len(vals)

def best_split(X, y, max_features=6):
    best_mse = float('inf'); best_f = best_t = None
    n_feat = len(X[0])
    feats = random.sample(range(n_feat), min(max_features, n_feat))
    for f in feats:
        vals = sorted(set(row[f] for row in X))
        thresholds = [(vals[i]+vals[i+1])/2 for i in range(len(vals)-1)]
        thresholds = thresholds[::max(1,len(thresholds)//20)]  # sample thresholds
        for t in thresholds:
            ly = [y[i] for i,row in enumerate(X) if row[f] <= t]
            ry = [y[i] for i,row in enumerate(X) if row[f] >  t]
            if len(ly)<3 or len(ry)<3: continue
            score = (len(ly)*mse(ly) + len(ry)*mse(ry)) / len(y)
            if score < best_mse:
                best_mse=score; best_f=f; best_t=t
    return best_f, best_t

def build_tree(X, y, depth=0, max_depth=8, min_samples=12):
    node = Node()
    if depth >= max_depth or len(y) <= min_samples:
        node.value = sum(y)/len(y); return node
    f, t = best_split(X, y)
    if f is None:
        node.value = sum(y)/len(y); return node
    left_mask  = [i for i,row in enumerate(X) if row[f] <= t]
    right_mask = [i for i,row in enumerate(X) if row[f] >  t]
    node.feature=f; node.threshold=t
    node.left  = build_tree([X[i] for i in left_mask],  [y[i] for i in left_mask],  depth+1, max_depth, min_samples)
    node.right = build_tree([X[i] for i in right_mask], [y[i] for i in right_mask], depth+1, max_depth, min_samples)
    return node

def predict_tree(node, x):
    if node.value is not None: return node.value
    if x[node.feature] <= node.threshold: return predict_tree(node.left, x)
    return predict_tree(node.right, x)

def tree_to_dict(node):
    if node.value is not None: return {'value': round(node.value,4)}
    return {'feature':node.feature,'threshold':node.threshold,
            'left':tree_to_dict(node.left),'right':tree_to_dict(node.right)}

def dict_to_tree(d):
    node = Node()
    if 'value' in d: node.value=d['value']; return node
    node.feature=d['feature']; node.threshold=d['threshold']
    node.left=dict_to_tree(d['left']); node.right=dict_to_tree(d['right'])
    return node

# Train Random Forest (10 trees)
print("  Building 10 trees...")
N_TREES = 10
forest = []
for i in range(N_TREES):
    # Bootstrap sample
    bag_idx = [random.randint(0, len(X_train)-1) for _ in range(len(X_train))]
    Xb = [X_train[j] for j in bag_idx]
    yb = [y_train[j] for j in bag_idx]
    tree = build_tree(Xb, yb, max_depth=8, min_samples=12)
    forest.append(tree)
    print(f"    Tree {i+1}/10 done")

def forest_predict(forest, x):
    return sum(predict_tree(t,x) for t in forest) / len(forest)

# ── 3. Evaluate ───────────────────────────────────────────────────────────
print("📊 Evaluating model...")
preds = [forest_predict(forest, x) for x in X_test]

def rmse(actual, predicted):
    return math.sqrt(sum((a-p)**2 for a,p in zip(actual,predicted))/len(actual))

def mae(actual, predicted):
    return sum(abs(a-p) for a,p in zip(actual,predicted))/len(actual)

def r2(actual, predicted):
    mean_a = sum(actual)/len(actual)
    ss_tot = sum((a-mean_a)**2 for a in actual)
    ss_res = sum((a-p)**2 for a,p in zip(actual,predicted))
    return 1 - ss_res/ss_tot if ss_tot else 0

RMSE = round(rmse(y_test, preds), 2)
MAE  = round(mae(y_test, preds), 2)
R2   = round(r2(y_test, preds), 4)

print(f"  R² Score : {R2}")
print(f"  RMSE     : {RMSE}")
print(f"  MAE      : {MAE}")

metrics = {
    'r2': R2, 'rmse': RMSE, 'mae': MAE,
    'train_samples': len(X_train),
    'test_samples' : len(X_test),
    'n_trees': N_TREES,
    'algorithm': 'Random Forest Regressor (pure Python)',
    'features': FEATURES
}
with open('metrics.json','w') as f:
    json.dump(metrics, f, indent=2)
print("  ✅ metrics.json saved")

# ── 4. Save Model ─────────────────────────────────────────────────────────
print("💾 Saving model...")
model_data = [tree_to_dict(t) for t in forest]
with open('model_weights.json','w') as f:
    json.dump(model_data, f)
print("  ✅ model_weights.json saved")
print("\n🎉 Training complete! Now run: python app.py")
