import { useState } from 'react'

function SignalList({ signals }) {
  if (!signals || !signals.length) {
    return <p className="mt-4 text-sm text-slate-500">No known heuristic pattern matches.</p>
  }
  return (
    <ul className="mt-4 flex flex-wrap gap-2" aria-label="Heuristic pattern matches">
      {signals.map((signal) => (
        <li key={signal} className="rounded-full border border-coral/40 bg-coral/10 px-3 py-1 text-sm text-coral">
          {signal}
        </li>
      ))}
    </ul>
  )
}

function EnglishResultCard({ result }) {
  const percentage = Math.round(result.jailbreak_probability * 100)
  const riskStyles = { low: 'bg-mint text-ink', medium: 'bg-amber text-ink', high: 'bg-coral text-white' }
  return (
    <section className="panel" aria-live="polite" aria-label="Analysis result">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="eyebrow">Calibrated assessment</p>
          <h2 className="mt-2 font-display text-3xl capitalize text-ink">{result.label}</h2>
        </div>
        <span className={'rounded-full px-3 py-1 text-sm font-semibold uppercase tracking-wider ' + riskStyles[result.risk_level]}>
          {result.risk_level} risk
        </span>
      </div>
      <div className="mt-8">
        <div className="flex items-end justify-between">
          <span className="text-sm font-medium text-slate-500">Calibrated jailbreak probability</span>
          <strong className="font-display text-4xl text-ink">{percentage}%</strong>
        </div>
        <div className="mt-3 h-3 overflow-hidden rounded-full bg-slate-200" aria-label={percentage + '% calibrated jailbreak probability'} role="meter" aria-valuemin="0" aria-valuemax="100" aria-valuenow={percentage}>
          <div className="h-full rounded-full bg-coral transition-all duration-500" style={{ width: percentage + '%' }} />
        </div>
      </div>
      <div className="mt-8 grid gap-3 border-t border-slate-200 pt-5 text-sm text-slate-600 sm:grid-cols-2">
        <span>Model: <strong className="text-ink">{result.model_version}</strong></span>
        <span>{result.calibrated ? 'Temperature calibrated' : 'Calibration unavailable'}</span>
        {result.input_truncated && (
          <span className="rounded-lg border border-amber bg-amber/30 px-3 py-2 text-ink sm:col-span-2">
            Prompt exceeded 256 model tokens and was truncated before inference.
          </span>
        )}
      </div>
      <div className="mt-6">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-500">Heuristic pattern matches</h3>
        <p className="mt-2 text-xs text-slate-500">These deterministic patterns are separate context, not model attribution.</p>
        <SignalList signals={result.heuristic_signals} />
      </div>
    </section>
  )
}

function AssessmentCard({ title, analysis, inputTruncated, isTranslation = false }) {
  const percentage = Math.round(analysis.jailbreak_probability * 100)
  const riskStyles = { low: 'bg-mint text-ink', medium: 'bg-amber text-ink', high: 'bg-coral text-white' }
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs font-bold uppercase tracking-wider text-slate-500">{title}</p>
          <h3 className="mt-1 font-display text-2xl capitalize text-ink">{analysis.label}</h3>
        </div>
        <span className={'rounded-full px-2.5 py-0.5 text-xs font-semibold uppercase tracking-wider ' + riskStyles[analysis.risk_level]}>
          {analysis.risk_level} risk
        </span>
      </div>
      <div className="mt-4">
        <div className="flex items-end justify-between text-xs">
          <span className="text-slate-500">{isTranslation ? 'Uncalibrated risk probability' : 'Calibrated probability'}</span>
          <strong className="font-display text-xl text-ink">{percentage}%</strong>
        </div>
        <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-100">
          <div className="h-full rounded-full bg-coral" style={{ width: percentage + '%' }} />
        </div>
      </div>
      <div className="mt-4 flex flex-wrap gap-2 text-xs text-slate-600">
        <span>Model: <strong>{analysis.model_version}</strong></span>
        <span>·</span>
        <span>{analysis.calibrated ? 'Calibrated' : 'Uncalibrated'}</span>
        {inputTruncated && (
          <span className="w-full text-amber font-medium">Input truncated (dual window evaluated)</span>
        )}
      </div>
      {analysis.heuristic_signals && analysis.heuristic_signals.length > 0 && (
        <div className="mt-3">
          <SignalList signals={analysis.heuristic_signals} />
        </div>
      )}
    </div>
  )
}

export function MultilingualResultView({ result }) {
  const riskStyles = { low: 'bg-mint text-ink', medium: 'bg-amber text-ink', high: 'bg-coral text-white' }
  const langNames = { sw: 'Swahili', ha: 'Hausa', bn: 'Bengali' }
  const modeNames = { multilingual: 'Multilingual model', translation: 'Translate to English', compare: 'Comparative evaluation' }

  return (
    <section className="panel" aria-live="polite" aria-label="Multilingual analysis result">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="eyebrow">{langNames[result.language] || result.language} · {modeNames[result.mode] || result.mode}</p>
          <h2 className="mt-2 font-display text-3xl capitalize text-ink">Conservative decision: {result.decision}</h2>
        </div>
        <span className={'rounded-full px-3 py-1 text-sm font-semibold uppercase tracking-wider ' + riskStyles[result.risk_level]}>
          {result.risk_level} risk
        </span>
      </div>

      {result.warnings && result.warnings.length > 0 && (
        <div className="mt-5 space-y-2" aria-live="assertive" role="status">
          {result.warnings.map((w, idx) => (
            <p key={idx} className="rounded-lg border border-amber/40 bg-amber/20 px-3.5 py-2 text-xs text-ink font-medium">
              ⚠ {w}
            </p>
          ))}
        </div>
      )}

      <div className="mt-7 grid gap-4 sm:grid-cols-2">
        {result.multilingual_assessment && (
          <AssessmentCard
            title="Direct Multilingual Model"
            analysis={result.multilingual_assessment}
            inputTruncated={result.multilingual_assessment.input_truncated}
            isTranslation={false}
          />
        )}
        {result.translation_assessment && (
          <AssessmentCard
            title={`Translation-assisted (${result.translation_assessment.translation_model})`}
            analysis={result.translation_assessment.analysis}
            inputTruncated={result.translation_assessment.translation_input_truncated}
            isTranslation={true}
          />
        )}
      </div>
    </section>
  )
}

export default function Analyzer({
  prompt,
  setPrompt,
  language,
  setLanguage,
  mode,
  setMode,
  onSubmit,
  result,
  isLoading,
  error,
}) {
  const isMultilingual = language !== 'en'

  return (
    <section className="panel">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="eyebrow">Live analysis</p>
          <h2 className="mt-1 font-display text-3xl text-ink">Test a prompt</h2>
        </div>
        <span className="text-xs text-slate-500 font-mono">
          {isMultilingual ? 'POST /api/analyze-multilingual' : 'POST /api/analyze'}
        </span>
      </div>

      <form className="mt-6" onSubmit={onSubmit}>
        <div className="grid gap-3 sm:grid-cols-2">
          <div>
            <label htmlFor="language-select" className="block text-xs font-semibold uppercase tracking-wider text-slate-500">
              Language
            </label>
            <select
              id="language-select"
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="mt-1.5 w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm text-ink outline-none transition focus:border-ink focus:ring-2 focus:ring-mint/60"
            >
              <option value="en">English</option>
              <option value="sw">Swahili (sw)</option>
              <option value="ha">Hausa (ha)</option>
              <option value="bn">Bengali (bn)</option>
            </select>
          </div>

          {isMultilingual && (
            <div>
              <label htmlFor="mode-select" className="block text-xs font-semibold uppercase tracking-wider text-slate-500">
                Evaluation mode
              </label>
              <select
                id="mode-select"
                value={mode}
                onChange={(e) => setMode(e.target.value)}
                className="mt-1.5 w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm text-ink outline-none transition focus:border-ink focus:ring-2 focus:ring-mint/60"
              >
                <option value="compare">Compare both (conservative)</option>
                <option value="multilingual">Multilingual model</option>
                <option value="translation">Translate to English</option>
              </select>
            </div>
          )}
        </div>

        <div className="mt-4">
          <label className="sr-only" htmlFor="prompt">Prompt to analyze</label>
          <textarea
            id="prompt"
            className="min-h-48 w-full resize-y rounded-xl border border-slate-300 bg-white p-4 text-base leading-7 text-ink outline-none transition placeholder:text-slate-400 focus:border-ink focus:ring-4 focus:ring-mint/60"
            maxLength={5000}
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder={
              language === 'bn'
                ? 'বাংলা প্রম্পট এখানে লিখুন...'
                : language === 'sw'
                ? 'Andika prompt kwa Kiswahili hapa...'
                : language === 'ha'
                ? 'Rubuta tambaya a Hausa anan...'
                : 'Paste a prompt to inspect calibrated probability, risk bands, and heuristic signals...'
            }
          />
        </div>

        <div className="mt-4 flex flex-col justify-between gap-3 sm:flex-row sm:items-center">
          <span className="text-xs text-slate-500">{prompt.length} / 5000 characters</span>
          <button className="primary-button" disabled={isLoading} type="submit">
            {isLoading ? 'Analyzing…' : 'Run evaluation'}
          </button>
        </div>
      </form>

      {error && <p className="mt-5 rounded-lg border border-coral/40 bg-coral/10 px-4 py-3 text-sm text-coral" role="alert">{error}</p>}
      {result && (
        <div className="mt-6">
          {result.multilingual_assessment || result.translation_assessment ? (
            <MultilingualResultView result={result} />
          ) : (
            <EnglishResultCard result={result} />
          )}
        </div>
      )}
    </section>
  )
}
