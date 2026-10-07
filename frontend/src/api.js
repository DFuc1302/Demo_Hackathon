const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
const REQUEST_TIMEOUT_MS = 15000

function errorMessage(payload) {
  const detail = payload.detail
  if (Array.isArray(detail)) {
    const messages = detail.map((item) => typeof item === 'string' ? item : item && item.msg).filter(Boolean)
    if (messages.length) return messages.join('; ')
  }
  if (typeof detail === 'string') return detail
  return 'The API request could not be completed.'
}

async function request(path, options = {}) {
  const timeoutController = new AbortController()
  const callerSignal = options.signal
  let timedOut = false
  const timer = setTimeout(() => { timedOut = true; timeoutController.abort() }, REQUEST_TIMEOUT_MS)
  const abort = () => timeoutController.abort()
  if (callerSignal?.aborted) timeoutController.abort()
  else callerSignal?.addEventListener('abort', abort, { once: true })
  try {
    const response = await fetch(API_BASE_URL + path, { ...options, signal: timeoutController.signal })
    const payload = await response.json().catch(() => ({}))
    if (!response.ok) throw new Error(errorMessage(payload))
    return payload
  } catch (error) {
    if (timedOut && error.name === 'AbortError') throw new Error('The API request timed out after 15 seconds.')
    throw error
  } finally {
    clearTimeout(timer)
    callerSignal?.removeEventListener('abort', abort)
  }
}

export function analyzePrompt(prompt, signal) {
  return request('/api/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ prompt }), signal })
}

export function getModelInfo(signal) {
  return request('/api/model-info', { signal })
}

export function getRedTeamCases(signal) {
  return request('/api/redteam/cases', { signal })
}

export function runRedTeam(caseIds, signal) {
  return request('/api/redteam/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ case_ids: caseIds }), signal })
}

export function validateCompletion(completion, systemPromptSnippets = null, signal) {
  return request('/api/guardrail/validate-completion', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ completion, system_prompt_snippets: systemPromptSnippets }),
    signal,
  })
}
export function analyzeMultilingual(prompt, language, mode = 'compare', signal) {
  return request('/api/analyze-multilingual', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, language, mode }),
    signal,
  })
}

export function getMultilingualModelInfo(signal) {
  return request('/api/multilingual/model-info', { signal })
}
