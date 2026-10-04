const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

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
  const response = await fetch(API_BASE_URL + path, options)
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(errorMessage(payload))
  return payload
}

export function analyzePrompt(prompt) {
  return request('/api/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ prompt }) })
}

export function getRedTeamCases() {
  return request('/api/redteam/cases')
}

export function runRedTeam(caseIds) {
  return request('/api/redteam/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ case_ids: caseIds }) })
}
