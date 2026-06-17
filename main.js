/* TransitIQ v3 — main.js */

// ── Toggle helpers ────────────────────────────────────────────────────────
function initToggle(cid, hid) {
  const c = document.getElementById(cid), h = document.getElementById(hid);
  c.querySelectorAll('[data-value]').forEach(btn => {
    btn.addEventListener('click', () => {
      c.querySelectorAll('[data-value]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      h.value = btn.dataset.value;
    });
  });
}
initToggle('modeSelector','mode');
initToggle('routeToggle','route_type');
initToggle('payToggle','payment');
initToggle('dayToggle','day_type');

// Default date/time
document.getElementById('journey_date').value = new Date().toISOString().split('T')[0];
document.getElementById('departure_time').value = '08:00';

// ── Counter ───────────────────────────────────────────────────────────────
let predCount = 0;
function animateCounter(el, target) {
  let cur = parseInt(el.textContent)||0;
  const step = Math.max(1, Math.ceil((target-cur)/20));
  const t = setInterval(()=>{
    cur = Math.min(cur+step, target);
    el.textContent = cur;
    if(cur>=target) clearInterval(t);
  }, 40);
}

// ── Chart instances ───────────────────────────────────────────────────────
const charts = {};
const THEME = {
  grid: 'rgba(255,255,255,0.05)',
  tick: '#64748B',
  font: 'DM Mono',
  tooltip: { bg:'#0D1421', border:'rgba(255,255,255,0.1)', title:'#E2E8F0' }
};
function baseOpts(extra={}) {
  return {
    responsive:true, maintainAspectRatio:false,
    plugins:{ legend:{display:false},
      tooltip:{ backgroundColor:THEME.tooltip.bg, borderColor:THEME.tooltip.border,
        borderWidth:1, titleColor:THEME.tooltip.title, ...extra.tooltip }},
    scales:{
      x:{ grid:{color:THEME.grid,borderColor:'transparent'},
          ticks:{color:THEME.tick, font:{family:THEME.font,size:10}} },
      y:{ grid:{color:THEME.grid,borderColor:'transparent'},
          ticks:{color:THEME.tick, font:{family:THEME.font,size:10}} }
    }, animation:{duration:900,easing:'easeInOutQuart'}, ...extra
  };
}

function makeGradient(ctx, color, h) {
  const g = ctx.createLinearGradient(0,0,0,h||220);
  g.addColorStop(0, color+'55'); g.addColorStop(1, color+'00');
  return g;
}

function buildChart(id, type, labels, datasets, extraOpts={}) {
  const canvas = document.getElementById(id);
  const ctx = canvas.getContext('2d');
  if(charts[id]) charts[id].destroy();
  charts[id] = new Chart(ctx, { type, data:{labels,datasets}, options:baseOpts(extraOpts) });
}

// ── Animate KPI numbers ───────────────────────────────────────────────────
function animVal(el, target, pre='', suf='') {
  let cur=0;
  const step = Math.max(1, Math.ceil(target/40));
  const t = setInterval(()=>{
    cur = Math.min(cur+step, target);
    el.textContent = pre + cur.toLocaleString() + suf;
    if(cur>=target) clearInterval(t);
  }, 28);
}

// ── Retrain ───────────────────────────────────────────────────────────────
async function retrain() {
  const btn = document.querySelector('.retrain-btn');
  btn.textContent = '⏳ Retraining...'; btn.disabled = true;
  try {
    const r = await fetch('/retrain', {method:'POST'});
    const d = await r.json();
    btn.textContent = `✅ R²=${d.test_r2}`;
    setTimeout(()=>{ btn.textContent='🔄 Retrain'; btn.disabled=false; }, 3000);
  } catch(e) {
    btn.textContent='🔄 Retrain'; btn.disabled=false;
  }
}

// ── Form submit ───────────────────────────────────────────────────────────
document.getElementById('predictForm').addEventListener('submit', async e => {
  e.preventDefault();
  const payload = {
    source:         document.getElementById('source').value,
    destination:    document.getElementById('dest').value,
    journey_date:   document.getElementById('journey_date').value,
    departure_time: document.getElementById('departure_time').value,
    mode:           document.getElementById('mode').value,
    travel_class:   document.getElementById('travel_class').value,
    ticket_price:   document.getElementById('ticket_price').value,
    travel_time:    document.getElementById('travel_time').value,
    route_type:     document.getElementById('route_type').value,
    payment:        document.getElementById('payment').value,
    day_type:       document.getElementById('day_type').value,
    is_holiday:     document.getElementById('is_holiday').checked ? 1 : 0
  };

  document.getElementById('loader').style.display = 'flex';

  try {
    const res  = await fetch('/predict', {method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    const data = await res.json();
    if(data.error){ alert('Error: '+data.error); return; }

    // Show sections
    ['kpiRow','charts','metrics','recoCard'].forEach(id=>{
      document.getElementById(id).style.display = id==='charts'?'grid':'block';
    });
    document.getElementById('kpiRow').style.display = 'grid';

    // KPIs
    animVal(document.getElementById('kDemand'), data.demand);
    animVal(document.getElementById('kOcc'),    Math.round(data.occupancy),'','%');
    animVal(document.getElementById('kRev'),    Math.round(data.revenue_estimate),'₹');
    document.getElementById('kMode').textContent   = data.preferred_mode;
    document.getElementById('kStatus').textContent = `Demand: ${data.status}`;

    setTimeout(()=>{
      document.getElementById('kDemandBar').style.width = Math.min(100,(data.demand/500)*100)+'%';
      document.getElementById('kOccBar').style.width    = data.occupancy+'%';
    },250);

    // ── Chart 1: Hourly Demand (line) ──
    const ctx1 = document.getElementById('cTrend').getContext('2d');
    const g1   = makeGradient(ctx1,'#00D4FF',240);
    buildChart('cTrend','line',
      data.trend_hours.map(h=>`${h}:00`),
      [{data:data.trend_demand, borderColor:'#00D4FF', backgroundColor:g1,
        borderWidth:2.5, pointRadius:4, pointBackgroundColor:'#00D4FF',
        pointHoverRadius:7, fill:true, tension:0.4}],
      {tooltip:{callbacks:{label:c=>` ${c.parsed.y} passengers`}}}
    );

    // ── Chart 2: Occupancy % (area) ──
    const ctx2 = document.getElementById('cOcc').getContext('2d');
    const g2   = makeGradient(ctx2,'#10B981',220);
    buildChart('cOcc','line',
      data.trend_hours.map(h=>`${h}:00`),
      [{data:data.occ_trend, borderColor:'#10B981', backgroundColor:g2,
        borderWidth:2, pointRadius:3, fill:true, tension:0.4}],
      {tooltip:{callbacks:{label:c=>` ${c.parsed.y}% occupancy`}},
       scales:{y:{min:0,max:100,
         grid:{color:'rgba(255,255,255,0.05)',borderColor:'transparent'},
         ticks:{color:'#64748B',font:{family:'DM Mono',size:10},callback:v=>v+'%'}}}}
    );

    // ── Chart 3: Mode Comparison (bar) ──
    const modes = Object.keys(data.mode_demands);
    const mVals = Object.values(data.mode_demands);
    buildChart('cMode','bar', modes,
      [{data:mVals,
        backgroundColor:['rgba(0,212,255,.2)','rgba(124,58,237,.2)','rgba(16,185,129,.2)'],
        borderColor:['#00D4FF','#7C3AED','#10B981'],
        borderWidth:2, borderRadius:10,
        hoverBackgroundColor:['rgba(0,212,255,.4)','rgba(124,58,237,.4)','rgba(16,185,129,.4)']}],
      {tooltip:{callbacks:{label:c=>` ${c.parsed.y} passengers`}}}
    );

    // ── Chart 4: Price Sensitivity (line) ──
    const ctx4 = document.getElementById('cPrice').getContext('2d');
    const g4   = makeGradient(ctx4,'#F59E0B',240);
    buildChart('cPrice','line',
      data.price_labels,
      [{data:data.price_sens, borderColor:'#F59E0B', backgroundColor:g4,
        borderWidth:2.5, pointRadius:5, pointBackgroundColor:'#F59E0B',
        fill:true, tension:0.35}],
      {tooltip:{callbacks:{label:c=>` ${c.parsed.y} passengers`}}}
    );

    // ── Metrics ──
    const m = data.metrics;
    document.getElementById('mTrain').textContent  = m.train_samples.toLocaleString()+' records';
    document.getElementById('mR2').textContent     = m.test_r2;
    document.getElementById('mRMSE').textContent   = m.test_rmse;
    document.getElementById('mMAE').textContent    = m.test_mae;
    document.getElementById('mEst').textContent    = m.n_estimators+' stumps';
    document.getElementById('mTrainR2').textContent= m.train_r2;
    document.getElementById('mLR').textContent     = m.learning_rate;

    // ── Recommendations ──
    const rl = document.getElementById('recoList');
    rl.innerHTML = '';
    data.recommendation.forEach(tip=>{
      const d = document.createElement('div');
      d.className = 'reco-item'; d.textContent = tip;
      rl.appendChild(d);
    });

    // Counter
    predCount++;
    animateCounter(document.getElementById('liveCount'), predCount);

    // Scroll to results
    document.getElementById('kpiRow').scrollIntoView({behavior:'smooth',block:'start'});

  } catch(err) {
    alert('Network error. Is Flask running? Check terminal for python app.py');
  } finally {
    document.getElementById('loader').style.display = 'none';
  }
});

// ── Nav highlight on scroll ───────────────────────────────────────────────
const navLinks = document.querySelectorAll('.nav-link');
window.addEventListener('scroll', ()=>{
  navLinks.forEach(link=>{
    const sec = document.querySelector(link.getAttribute('href'));
    if(sec && window.scrollY >= sec.offsetTop-120){
      navLinks.forEach(l=>l.classList.remove('active'));
      link.classList.add('active');
    }
  });
});
