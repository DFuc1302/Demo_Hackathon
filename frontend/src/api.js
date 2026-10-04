const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

async function request(path, options = {}) {
  const response = await fetch(API_BASE_URL + path, options)
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(payload.detail || 'The API request could not be completed.')
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
