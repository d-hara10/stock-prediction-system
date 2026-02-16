from arch import arch_model
from sklearn.metrics import mean_absolute_error, r2_score

from src.pipelines.volatility_pipeline import run_pipeline

tickers = ["AAPL", "TSLA", "JTEK"]

results = {}

for ticker in tickers:
    print(f"\n{'='*60}")
    print(f"Processing {ticker}")
    print('='*60)

    # Updated: now unpacks 8 values (added volatility_context)
    model, X_test, y_test, y_pred, df, metadata, retrained, vol_context = run_pipeline(ticker)
    
    # Show retrain status
    if retrained:
        print(f"Model retrained")
    else:
        trained_date = metadata.get('trained_on', 'unknown')[:10]  # Just the date
        print(f"Using cached model (trained: {trained_date})")
    
    # Show volatility context
    print(f"\n Volatility Forecast:")
    print(f"Current:   {vol_context['current']:.4f}")
    print(f"Predicted: {vol_context['predicted']:.4f} ({vol_context['change_pct']:+.1f}%)")
    print(f"Confidence: {vol_context['confidence'].upper()} (R²={vol_context['r2_score']:.3f})")
    
    # Random Forest metrics
    rf_mae = mean_absolute_error(y_test, y_pred)
    rf_r2 = r2_score(y_test, y_pred)

    # GARCH baseline
    garch_returns = df["Return"].dropna() * 100
    
    garch = arch_model(garch_returns, vol="Garch", p=1, q=1)
    garch_fit = garch.fit(disp="off")

    garch_forecast = garch_fit.forecast(horizon=len(y_test))
    garch_vol = garch_forecast.variance.values[-1, :] ** 0.5 / 100

    garch_mae = mean_absolute_error(y_test, garch_vol)

    # Show model performance
    print(f"\n Model Performance:")
    print(f"RF MAE: {rf_mae:.6f}")
    print(f"GARCH MAE: {garch_mae:.6f}")
    print(f"RF beats GARCH: {'✅ YES' if rf_mae < garch_mae else '❌ NO'}")
    print(f"R² Score:  {rf_r2:.4f}")

    results[ticker] = {
        "rf_mae": rf_mae,
        "rf_r2": rf_r2,
        "garch_mae": garch_mae,
        "vol_context": vol_context,
        "retrained": retrained
    }

print(f"\n{'='*60}")
print("SUMMARY")
print('='*60)

for ticker, metrics in results.items():
    vc = metrics['vol_context']
    print(f"\n{ticker}:")
    print(f"Volatility: {vc['current']:.4f} → {vc['predicted']:.4f} ({vc['change_pct']:+.1f}%)")
    print(f"Confidence: {vc['confidence']} (R²={vc['r2_score']:.3f}, MAE={vc['mae']:.6f})")
    print(f"Beats GARCH: {'✅' if metrics['rf_mae'] < metrics['garch_mae'] else '❌'}")
    print(f"Retrained: {'Yes' if metrics['retrained'] else 'No'}")


