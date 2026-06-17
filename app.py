"""
TransitIQ v3 — ML Transport Demand Prediction
✅ Real Gradient Boosting (pure Python)
✅ model.pkl save/load
✅ R², MAE, RMSE metrics
✅ Occupancy prediction
✅ Chart.js data endpoints
✅ Modern dashboard API
"""

from flask import Flask, render_template, request, jsonify
import random, math, json, os, pickle
from datetime import datetime

app = Flask(__name__)

# ══════════════════════════════════════════════════════════════
#  PURE PYTHON ML ENGINE
# ══════════════════════════════════════════════════════════════

class DecisionStump:
    def __init__(self):
        self.feature = self.threshold = None
        self.left_val = self.right_val = 0.0

    def fit(self, X, residuals):
        best_loss = float('inf')
        for f in range(len(X[0])):
            vals = sorted(set(row[f] for row in X))
            thresholds = [(vals[i]+vals[i+1])/2 for i in range(len(vals)-1)]
            for t in thresholds:
                L = [residuals[i] for i,r in enumerate(X) if r[f] <= t]
                R = [residuals[i] for i,r in enumerate(X) if r[f] >  t]
                if not L or not R: continue
                lv, rv = sum(L)/len(L), sum(R)/len(R)
                loss = sum((v-lv)**2 for v in L) + sum((v-rv)**2 for v in R)
                if loss < best_loss:
                    best_loss, self.feature, self.threshold = loss, f, t
                    self.left_val, self.right_val = lv, rv

    def predict_one(self, row):
        if self.feature is None: return 0.0
        return self.left_val if row[self.feature] <= self.threshold else self.right_val


class GBRegressor:
    """Gradient Boosting Regressor — real training, real metrics."""
    def __init__(self, n_estimators=50, lr=0.15):
        self.n_estimators = n_estimators
        self.lr = lr
        self.stumps = []
        self.base = 0.0
        self.metrics = {}

    def fit(self, X, y):
        self.base = sum(y)/len(y)
        preds = [self.base]*len(y)
        for _ in range(self.n_estimators):
            res = [y[i]-preds[i] for i in range(len(y))]
            s = DecisionStump(); s.fit(X, res)
            self.stumps.append(s)
            for i,row in enumerate(X):
                preds[i] += self.lr * s.predict_one(row)
        self._calc_metrics(y, preds, tag='train')
        return self

    def predict(self, row):
        p = self.base
        for s in self.stumps: p += self.lr * s.predict_one(row)
        return max(10, min(500, int(round(p))))

    def predict_batch(self, X):
        return [self.predict(row) for row in X]

    def _calc_metrics(self, y, preds, tag='train'):
        n = len(y)
        mean_y = sum(y)/n
        ss_res = sum((y[i]-preds[i])**2 for i in range(n))
        ss_tot = sum((yi-mean_y)**2 for yi in y)
        r2   = round(1 - ss_res/ss_tot, 4) if ss_tot else 1.0
        rmse = round(math.sqrt(ss_res/n), 2)
        mae  = round(sum(abs(y[i]-preds[i]) for i in range(n))/n, 2)
        self.metrics[tag] = {'r2': r2, 'rmse': rmse, 'mae': mae}

    def evaluate(self, X, y):
        preds = self.predict_batch(X)
        self._calc_metrics(y, preds, tag='test')
        return self.metrics['test']


# ══════════════════════════════════════════════════════════════
#  ENCODINGS
# ══════════════════════════════════════════════════════════════
MODE_ENC  = {'Bus':0,'Train':1,'Airline':2}
ROUTE_ENC = {'Fastest':0,'Easiest':1}
PAY_ENC   = {'Cash':0,'UPI':1,'Card':2}
CLASS_ENC = {'Economy':0,'Business':1,'First Class':2}

def encode_row(price, ttime, day, holiday, hour, mode, route, pay, cls):
    return [price, ttime, day, holiday, hour,
            MODE_ENC.get(mode,0), ROUTE_ENC.get(route,0),
            PAY_ENC.get(pay,0),   CLASS_ENC.get(cls,0)]


# ══════════════════════════════════════════════════════════════
#  DATASET  (We Use 2000 records)
# ══════════════════════════════════════════════════════════════
def generate_dataset(n=2000, seed=42):
    random.seed(seed)
    modes   = ['Bus','Train','Airline']
    routes  = ['Fastest','Easiest']
    pays    = ['Cash','UPI','Card']
    classes = ['Economy','Business','First Class']
    X, y = [], []
    records = []   # store raw dicts for CSV export
    for i in range(n):
        price   = random.randint(100,10000)
        ttime   = random.randint(30,720)
        day     = random.randint(0,1)
        holiday = random.randint(0,1)
        hour    = random.randint(0,23)
        mode    = random.choice(modes)
        route   = random.choice(routes)
        pay     = random.choice(pays)
        cls     = random.choice(classes)
        # Ground truth demand
        d = 250 - price*0.012 + day*35 + holiday*65 - ttime*0.08
        if 7<=hour<=9 or 17<=hour<=20: d+=60
        elif hour<=5: d-=40
        d += {'Bus':10,'Train':25,'Airline':15}[mode]
        d += {'Economy':20,'Business':-10,'First Class':-20}[cls]
        if route=='Fastest': d+=10
        if pay=='UPI': d+=8
        d += random.gauss(0,18)
        d = max(10, min(500, int(d)))
        X.append(encode_row(price,ttime,day,holiday,hour,mode,route,pay,cls))
        y.append(d)
        records.append({'id':i+1,'ticket_price':price,'travel_time':ttime,
                        'day_type':day,'is_holiday':holiday,'dep_hour':hour,
                        'mode':mode,'route':route,'payment':pay,'class':cls,'demand':d})
    return X, y, records


# ══════════════════════════════════════════════════════════════
#  TRAIN OR LOAD model.pkl
# ══════════════════════════════════════════════════════════════
PKL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

def train_and_save():
    print("⏳ Generating dataset (2000 records)...")
    X, y, records = generate_dataset(2000)
    split = int(0.8*len(X))
    X_tr,X_te = X[:split], X[split:]
    y_tr,y_te = y[:split], y[split:]

    print("🤖 Training Gradient Boosting (50 estimators)...")
    m = GBRegressor(n_estimators=50, lr=0.15)
    m.fit(X_tr, y_tr)
    test_metrics = m.evaluate(X_te, y_te)

    bundle = {'model': m, 'records': records,
              'X': X, 'y': y,
              'train_metrics': m.metrics['train'],
              'test_metrics': test_metrics}
    with open(PKL_PATH, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"💾 model.pkl saved!  Test R²={test_metrics['r2']}  RMSE={test_metrics['rmse']}  MAE={test_metrics['mae']}")
    return bundle

if os.path.exists(PKL_PATH):
    print("📦 Loading model.pkl...")
    with open(PKL_PATH,'rb') as f:
        bundle = pickle.load(f)
    print("✅ Model loaded from pkl!")
else:
    bundle = train_and_save()

gbr      = bundle['model']
records  = bundle['records']
ALL_X    = bundle['X']
ALL_Y    = bundle['y']
TR_METRICS = bundle['train_metrics']
TE_METRICS = bundle['test_metrics']


# ══════════════════════════════════════════════════════════════
#  HELPER
# ══════════════════════════════════════════════════════════════
def get_status(occ):
    if occ>=80: return 'High'
    if occ>=50: return 'Moderate'
    return 'Low'

def get_tips(demand, occ, mode):
    tips = []
    if occ>=80:    tips.append(f"⚠️ High demand — add extra {mode} capacity to avoid overcrowding.")
    if occ<30:     tips.append("💡 Low demand period — consider promotional or discounted pricing.")
    if demand>300: tips.append("📈 Peak demand detected — implement dynamic surge pricing.")
    tips.append("✅ Pre-booking offers can improve utilization by 15–20%.")
    tips.append("🕐 Redistribute passengers to off-peak slots with targeted discounts.")
    return tips

def pref_mode(price, ttime):
    if price>4000: return 'Airline ✈️'
    if price>800 or ttime>180: return 'Train 🚆'
    return 'Bus 🚌'


# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html',
        train_r2   = TR_METRICS['r2'],
        test_r2    = TE_METRICS['r2'],
        test_rmse  = TE_METRICS['rmse'],
        test_mae   = TE_METRICS['mae'])

@app.route('/predict', methods=['POST'])
def predict():
    d = request.json
    try:
        day     = 1 if d['day_type']=='Weekend' else 0
        hol     = int(d['is_holiday'])
        hour    = datetime.strptime(d['departure_time'],'%H:%M').hour
        price   = float(d['ticket_price'])
        ttime   = float(d['travel_time'])

        row     = encode_row(price,ttime,day,hol,hour,
                             d['mode'],d['route_type'],d['payment'],d['travel_class'])
        demand  = gbr.predict(row)
        occ     = round(min(100,(demand/500)*100),1)

        # Hourly trend
        hours = list(range(5,24))
        trend = [gbr.predict(encode_row(price,ttime,day,hol,h,
                 d['mode'],d['route_type'],d['payment'],d['travel_class'])) for h in hours]

        # Occupancy trend
        occ_trend = [round(min(100,(v/500)*100),1) for v in trend]

        # Mode comparison
        mode_demands = {m: gbr.predict(encode_row(price,ttime,day,hol,hour,
                        m,d['route_type'],d['payment'],d['travel_class']))
                        for m in ['Bus','Train','Airline']}

        # Price sensitivity (vary price ±50%)
        prices     = [int(price*f) for f in [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.5,2.0]]
        price_sens = [gbr.predict(encode_row(p,ttime,day,hol,hour,
                      d['mode'],d['route_type'],d['payment'],d['travel_class']))
                      for p in prices]

        return jsonify({
            'demand'          : demand,
            'occupancy'       : occ,
            'preferred_mode'  : pref_mode(price,ttime),
            'revenue_estimate': round(demand*price,2),
            'status'          : get_status(occ),
            'recommendation'  : get_tips(demand,occ,d['mode']),
            'trend_hours'     : hours,
            'trend_demand'    : trend,
            'occ_trend'       : occ_trend,
            'mode_demands'    : mode_demands,
            'price_labels'    : [f'₹{p}' for p in prices],
            'price_sens'      : price_sens,
            'metrics': {
                'algorithm'    : 'Gradient Boosting Regressor',
                'n_estimators' : gbr.n_estimators,
                'learning_rate': gbr.lr,
                'train_r2'     : TR_METRICS['r2'],
                'train_rmse'   : TR_METRICS['rmse'],
                'train_mae'    : TR_METRICS['mae'],
                'test_r2'      : TE_METRICS['r2'],
                'test_rmse'    : TE_METRICS['rmse'],
                'test_mae'     : TE_METRICS['mae'],
                'train_samples': 1600,
                'test_samples' : 400,
            }
        })
    except Exception as e:
        return jsonify({'error':str(e)}), 400

@app.route('/stats')
def stats():
    avg_d  = round(sum(ALL_Y)/len(ALL_Y),1)
    mode_avgs = {}
    for m in ['Bus','Train','Airline']:
        vals = [records[i]['demand'] for i in range(len(records)) if records[i]['mode']==m]
        mode_avgs[m] = round(sum(vals)/len(vals),1) if vals else 0
    return jsonify({
        'total_records' : len(ALL_Y),
        'avg_demand'    : avg_d,
        'test_r2'       : TE_METRICS['r2'],
        'test_rmse'     : TE_METRICS['rmse'],
        'test_mae'      : TE_METRICS['mae'],
        'mode_avgs'     : mode_avgs,
    })

@app.route('/retrain', methods=['POST'])
def retrain():
    global bundle, gbr, records, ALL_X, ALL_Y, TR_METRICS, TE_METRICS
    if os.path.exists(PKL_PATH): os.remove(PKL_PATH)
    bundle     = train_and_save()
    gbr        = bundle['model']
    records    = bundle['records']
    ALL_X      = bundle['X']
    ALL_Y      = bundle['y']
    TR_METRICS = bundle['train_metrics']
    TE_METRICS = bundle['test_metrics']
    return jsonify({'status':'retrained', 'test_r2': TE_METRICS['r2']})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
