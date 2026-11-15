#!/usr/bin/env bash
# Solar Lens – one-command demo launcher
set -e
cd "$(dirname "$0")"

echo "═══════════════════════════════════════════════════"
echo "  ☀️  Solar Lens – Hackathon MVP"
echo "═══════════════════════════════════════════════════"
echo ""
echo "📦 Installing dependencies…"
pip install -q -r requirements.txt

echo ""
echo "📁 Creating artifacts directory…"
mkdir -p artifacts

echo ""
echo "🔧 Running training pipeline…"
python3 -c "
import sys; sys.path.insert(0, '.')
from src.ingest import load_all_data
from src.intervals import build_intervals
from src.features import add_features
from src.train import train_and_evaluate
from src.deviations import detect_deviations

data = load_all_data('.')
intervals = build_intervals(data)
intervals = add_features(intervals)

for pid in sorted(intervals['plant_id'].unique()):
    result = train_and_evaluate(intervals, pid)
    devs, band = detect_deviations(result['test_predictions'], result['val_residuals'])
    print(f'\nPlant {pid}:')
    print(f'  Champion: {result[\"champion_name\"]}')
    for name, m in result['metrics']['validation'].items():
        tag = ' <- champion' if name == result['champion_name'] else ''
        print(f'  {name}: Val MAE={m[\"mae\"]:.2f}  RMSE={m[\"rmse\"]:.2f}  R2={m[\"r2\"]:.4f}{tag}')
    t = result['metrics']['test']
    print(f'  TEST  MAE={t[\"mae\"]:.2f}  RMSE={t[\"rmse\"]:.2f}  R2={t[\"r2\"]:.4f}')
    print(f'  Tolerance band: [{band[\"lower\"]:.2f}, {band[\"upper\"]:.2f}]')
    print(f'  Deviations found: {len(devs)}')
"

echo ""
echo "✅ Pipeline complete!"
echo ""
echo "🚀 Launching Streamlit dashboard…"
echo "   Open http://localhost:8501 in your browser"
echo ""
streamlit run app/dashboard.py --server.headless true
