function Metric({ label, value }) {
  return (
    <span className="rounded-lg border border-slate-200 bg-paper px-3 py-2">
      <span className="block text-xs uppercase tracking-wider text-slate-500">{label}</span>
      <strong className="mt-1 block text-ink">{typeof value === 'number' ? value.toFixed(3) : value}</strong>
    </span>
  )
}

export default function ModelInfo({ modelInfo, multilingualInfo, isLoading, error }) {
  const multiMetrics = multilingualInfo?.metrics?.multilingual?.test
  const transMetrics = multilingualInfo?.metrics?.translation?.test
  const compMetrics = multilingualInfo?.metrics?.compare?.test

  return (
    <details className="panel" open>
      <summary className="cursor-pointer list-none">
        <p className="eyebrow">Detector provenance</p>
        <h2 className="mt-2 font-display text-3xl text-ink">Model and evaluation</h2>
      </summary>
      {isLoading && <p className="mt-5 text-sm text-slate-500" role="status">Loading model metadata…</p>}
      {error && (
        <p className="mt-5 rounded-lg border border-coral/40 bg-coral/10 px-4 py-3 text-sm text-coral" role="alert">
          Model information unavailable: {error}
        </p>
      )}
      {modelInfo && (
        <div className="mt-6 space-y-6 text-sm">
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">English V2 Release</h3>
            <div className="mt-2 grid gap-2 text-slate-600">
              <span>Base model: <strong className="text-ink">{modelInfo.base_model}</strong></span>
              <span>Dataset: <strong className="text-ink">{modelInfo.dataset}</strong></span>
              <span>Version: <strong className="text-ink">{modelInfo.model_version}</strong></span>
              <span>Calibration status: <strong className="text-ink">{modelInfo.calibrated ? 'Calibrated (temperature ' + modelInfo.temperature.toFixed(3) + ')' : 'Uncalibrated'}</strong></span>
            </div>
            {modelInfo.metrics?.test && (
              <div className="mt-3">
                <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">Held-out test split metrics</p>
                <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-4">
                  <Metric label="F1" value={modelInfo.metrics.test.f1} />
                  <Metric label="Recall" value={modelInfo.metrics.test.recall} />
                  <Metric label="Precision" value={modelInfo.metrics.test.precision} />
                  <Metric label="Brier score" value={modelInfo.metrics.test.brier_score} />
                </div>
              </div>
            )}
          </div>

          <div className="border-t border-slate-200 pt-5">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                Low-Resource Multilingual Pilot (Machine-translated benchmark)
              </h3>
              <span className={'rounded-full px-2.5 py-0.5 text-xs font-semibold ' + (multilingualInfo?.available ? 'bg-mint text-ink' : 'bg-slate-200 text-slate-600')}>
                {multilingualInfo?.available ? 'Installed & Active' : 'Not installed'}
              </span>
            </div>

            {multilingualInfo?.available ? (
              <div className="mt-3 space-y-4">
                <div className="grid gap-1.5 text-xs text-slate-600">
                  <span>Classifier: <strong className="text-ink">{multilingualInfo.classifier_model}</strong></span>
                  <span>Translator: <strong className="text-ink">{multilingualInfo.translation_model}</strong></span>
                  <span>Benchmark type: <strong className="text-ink">{multilingualInfo.benchmark_kind}</strong></span>
                </div>

                {multiMetrics && (
                  <div>
                    <p className="text-xs font-semibold text-slate-500">Held-out Test Performance by Language</p>
                    <div className="mt-2 overflow-x-auto">
                      <table className="w-full text-left text-xs text-slate-600">
                        <thead className="border-b border-slate-200 text-slate-400">
                          <tr>
                            <th className="py-1 font-semibold">Language</th>
                            <th className="py-1 font-semibold">Direct F1</th>
                            <th className="py-1 font-semibold">Direct Rec</th>
                            <th className="py-1 font-semibold">Trans Rec</th>
                            <th className="py-1 font-semibold">Compare Rec</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                          {['sw', 'ha', 'bn'].map((lang) => (
                            <tr key={lang}>
                              <td className="py-1.5 font-bold uppercase text-ink">{lang}</td>
                              <td className="py-1.5">{multiMetrics[lang]?.f1?.toFixed(3) || '—'}</td>
                              <td className="py-1.5">{multiMetrics[lang]?.recall?.toFixed(3) || '—'}</td>
                              <td className="py-1.5">{transMetrics?.[lang]?.recall?.toFixed(3) || '—'}</td>
                              <td className="py-1.5 font-semibold text-ink">{compMetrics?.[lang]?.recall?.toFixed(3) || '—'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="mt-3 rounded-xl border border-slate-200 bg-paper p-3.5 text-xs text-slate-600">
                <p>Multilingual models are optional and not currently installed.</p>
                <p className="mt-1 font-mono text-[11px] text-slate-500">
                  Run <code>python backend/scripts/train_multilingual_model.py</code> to enable.
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </details>
  )
}
