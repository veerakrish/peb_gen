import { useState } from 'react'
import ComponentForm from './components/ComponentForm'
import ConnectionMatrix from './components/ConnectionMatrix'
import LayoutPreview from './components/LayoutPreview'
import { preparePcb } from './api'
import { droneExampleProject } from './droneExample'
import './index.css'

const STEPS = ['Components & Pins', 'Netlist Connections', 'Layout Preview']

function StepBreadcrumb({ step }) {
  return (
    <ol className="mb-8 flex items-center gap-2 text-sm">
      {STEPS.map((label, i) => {
        const n = i + 1
        const active = n === step
        const done = n < step
        return (
          <li key={label} className="flex items-center gap-2">
            <span
              className={`flex h-7 w-7 items-center justify-center rounded-full border text-xs font-semibold ${
                active
                  ? 'border-emerald-400 bg-emerald-400 text-slate-900'
                  : done
                    ? 'border-emerald-500/60 text-emerald-400'
                    : 'border-slate-600 text-slate-500'
              }`}
            >
              {n}
            </span>
            <span className={active ? 'text-slate-100' : 'text-slate-500'}>{label}</span>
            {n !== STEPS.length && <span className="mx-2 text-slate-600">—</span>}
          </li>
        )
      })}
    </ol>
  )
}

export default function App() {
  const [step, setStep] = useState(1)
  const [board, setBoard] = useState({ width_mm: 50, height_mm: 50 })
  const [components, setComponents] = useState([])
  const [connections, setConnections] = useState([])
  const [result, setResult] = useState(null)
  const [sessionId, setSessionId] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  function loadDroneExample() {
    const example = droneExampleProject()
    setBoard(example.board)
    setComponents(example.components)
    setConnections(example.connections)
    setError(null)
  }

  function reset() {
    setStep(1)
    setBoard({ width_mm: 50, height_mm: 50 })
    setComponents([])
    setConnections([])
    setResult(null)
    setSessionId(null)
    setError(null)
  }

  async function handlePrepare() {
    setLoading(true)
    setError(null)
    const project = {
      board,
      components: components.map((c) => ({
        id: c.id,
        label: c.label,
        width_mm: c.width_mm,
        height_mm: c.height_mm,
        pins: c.pins,
      })),
      connections,
    }
    try {
      const response = await preparePcb(project)
      setResult(response)
      setSessionId(response.session_id)
      setStep(3)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-900 px-6 py-10 text-slate-100">
      <div className="mx-auto max-w-5xl">
        <div className="mb-6 flex items-center justify-between">
          <h1 className="text-2xl font-semibold">pcb_gen — Schematic to PCB</h1>
          {step === 1 && components.length === 0 && (
            <button
              type="button"
              onClick={loadDroneExample}
              className="rounded-lg border border-slate-600 px-3 py-1.5 text-sm text-slate-300 hover:bg-slate-800"
            >
              Load drone example
            </button>
          )}
          {(step > 1 || components.length > 0) && (
            <button
              type="button"
              onClick={reset}
              className="rounded-lg border border-slate-600 px-3 py-1.5 text-sm text-slate-300 hover:bg-slate-800"
            >
              Start over
            </button>
          )}
        </div>

        <StepBreadcrumb step={step} />

        {step === 1 && (
          <div>
            <ComponentForm board={board} onBoardChange={setBoard} components={components} onChange={setComponents} />
            <div className="mt-6 flex justify-end">
              <button
                type="button"
                disabled={components.length === 0}
                onClick={() => setStep(2)}
                className="rounded-lg bg-emerald-500 px-6 py-2.5 font-medium text-slate-900 hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Next: Connections
              </button>
            </div>
          </div>
        )}

        {step === 2 && (
          <div>
            <div className="mb-4">
              <button
                type="button"
                onClick={() => setStep(1)}
                className="text-sm text-slate-400 hover:text-slate-200"
              >
                ← Back to components
              </button>
            </div>
            <ConnectionMatrix
              components={components}
              connections={connections}
              onChange={setConnections}
              onPrepare={handlePrepare}
              loading={loading}
              error={error}
            />
          </div>
        )}

        {step === 3 && result && (
          <div>
            <div className="mb-4">
              <button
                type="button"
                onClick={() => setStep(2)}
                className="text-sm text-slate-400 hover:text-slate-200"
              >
                ← Back to connections
              </button>
            </div>
            <LayoutPreview
              project={{ board, components, connections }}
              result={result}
              sessionId={sessionId}
            />
          </div>
        )}
      </div>
    </div>
  )
}
