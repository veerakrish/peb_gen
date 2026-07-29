// Talks to the FastAPI backend (see webapp/backend/app/main.py). The Vite
// dev server proxies /api/* to http://localhost:8000 (vite.config.js), so
// relative paths work in both dev and behind a matching reverse proxy.

export async function preparePcb(project) {
  const res = await fetch('/api/prepare-pcb', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(project),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    const detail = body?.detail
      ? typeof body.detail === 'string'
        ? body.detail
        : JSON.stringify(body.detail)
      : `HTTP ${res.status}`
    throw new Error(detail)
  }
  return res.json()
}

export function kicadDownloadUrl(sessionId) {
  return `/api/download/kicad/${sessionId}`
}

export function gerberDownloadUrl(sessionId) {
  return `/api/download/gerber/${sessionId}`
}
