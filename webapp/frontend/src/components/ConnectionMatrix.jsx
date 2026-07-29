import { useState } from 'react'

function pinLabelText(component, pinNumber) {
  const pin = component?.pins.find((p) => p.number === pinNumber)
  return pin ? `${pin.label || '(unlabeled)'} (#${pin.number})` : ''
}

function EndpointPicker({ label, components, componentId, pinNumber, onChange }) {
  const component = components.find((c) => c.id === componentId)
  return (
    <div>
      <label className="block text-xs font-medium text-slate-400">{label}</label>
      <div className="mt-1 flex gap-2">
        <select
          className="w-40 rounded bg-slate-800 border border-slate-600 px-2 py-1 text-slate-100"
          value={componentId}
          onChange={(e) => onChange({ componentId: e.target.value, pinNumber: '' })}
        >
          <option value="">Component…</option>
          {components.map((c) => (
            <option key={c.id} value={c.id}>
              {c.label || '(unnamed)'}
            </option>
          ))}
        </select>
        <select
          className="w-40 rounded bg-slate-800 border border-slate-600 px-2 py-1 text-slate-100"
          value={pinNumber}
          onChange={(e) => onChange({ componentId, pinNumber: e.target.value })}
          disabled={!component}
        >
          <option value="">Pin…</option>
          {component?.pins.map((p) => (
            <option key={p.number} value={p.number}>
              {p.label || '(unlabeled)'} (#{p.number})
            </option>
          ))}
        </select>
      </div>
    </div>
  )
}

export default function ConnectionMatrix({ components, connections, onChange, onPrepare, loading, error }) {
  const [source, setSource] = useState({ componentId: '', pinNumber: '' })
  const [target, setTarget] = useState({ componentId: '', pinNumber: '' })

  const sourceComponent = components.find((c) => c.id === source.componentId)
  const targetComponent = components.find((c) => c.id === target.componentId)
  const suggestedNet =
    source.pinNumber && target.pinNumber
      ? sourceComponent?.pins.find((p) => p.number === source.pinNumber)?.label ||
        targetComponent?.pins.find((p) => p.number === target.pinNumber)?.label
      : null

  const canAdd =
    source.componentId && source.pinNumber && target.componentId && target.pinNumber &&
    !(source.componentId === target.componentId && source.pinNumber === target.pinNumber)

  function addConnection() {
    if (!canAdd) return
    onChange([
      ...connections,
      {
        source: { component_id: source.componentId, pin_number: source.pinNumber },
        target: { component_id: target.componentId, pin_number: target.pinNumber },
      },
    ])
    setSource({ componentId: source.componentId, pinNumber: '' })
    setTarget({ componentId: target.componentId, pinNumber: '' })
  }

  function removeConnection(index) {
    onChange(connections.filter((_, i) => i !== index))
  }

  function componentLabel(id) {
    return components.find((c) => c.id === id)?.label || id
  }

  return (
    <div>
      <div className="rounded-lg border border-slate-700 bg-slate-800/50 p-4">
        <div className="flex flex-wrap items-end gap-4">
          <EndpointPicker label="Source" components={components} {...source} onChange={setSource} />
          <div className="pb-2 text-slate-500">──▶</div>
          <EndpointPicker label="Target" components={components} {...target} onChange={setTarget} />
          <button
            type="button"
            disabled={!canAdd}
            onClick={addConnection}
            className="rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-4 py-2 text-emerald-300 hover:bg-emerald-500/20 disabled:cursor-not-allowed disabled:opacity-40"
          >
            + Add connection
          </button>
        </div>
        {suggestedNet && (
          <p className="mt-2 text-xs text-slate-400">
            Suggested net name: <span className="text-emerald-400">{suggestedNet}</span>
          </p>
        )}
      </div>

      <table className="mt-4 w-full text-sm">
        <thead>
          <tr className="text-left text-slate-400">
            <th className="pb-1 font-medium">Source</th>
            <th className="pb-1 font-medium" />
            <th className="pb-1 font-medium">Target</th>
            <th className="w-10 pb-1" />
          </tr>
        </thead>
        <tbody>
          {connections.map((conn, index) => {
            const srcComp = components.find((c) => c.id === conn.source.component_id)
            const tgtComp = components.find((c) => c.id === conn.target.component_id)
            return (
              <tr key={index} className="border-t border-slate-700/60">
                <td className="py-1.5 text-slate-200">
                  {componentLabel(conn.source.component_id)}.{pinLabelText(srcComp, conn.source.pin_number)}
                </td>
                <td className="py-1.5 text-slate-500">──▶</td>
                <td className="py-1.5 text-slate-200">
                  {componentLabel(conn.target.component_id)}.{pinLabelText(tgtComp, conn.target.pin_number)}
                </td>
                <td className="py-1.5 text-center">
                  <button
                    type="button"
                    onClick={() => removeConnection(index)}
                    className="text-slate-500 hover:text-red-400"
                    title="Remove connection"
                  >
                    ✕
                  </button>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>

      {error && (
        <div className="mt-4 rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-2 text-sm text-red-300">
          {error}
        </div>
      )}

      <button
        type="button"
        onClick={onPrepare}
        disabled={loading || connections.length === 0}
        className="mt-6 rounded-lg bg-emerald-500 px-6 py-2.5 font-medium text-slate-900 hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {loading ? 'Placing & routing…' : 'Prepare PCB'}
      </button>
    </div>
  )
}
