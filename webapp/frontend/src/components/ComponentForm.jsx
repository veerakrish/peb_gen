import PinTable from './PinTable'

let idCounter = 0
function nextId() {
  idCounter += 1
  return `comp_${idCounter}_${Date.now().toString(36)}`
}

export function makeComponent(label = '') {
  return {
    id: nextId(),
    label,
    width_mm: 20,
    height_mm: 20,
    pins: [
      { number: '1', label: 'VCC', function: 'power' },
      { number: '2', label: 'GND', function: 'ground' },
    ],
  }
}

export default function ComponentForm({ board, onBoardChange, components, onChange }) {
  function updateComponent(index, patch) {
    onChange(components.map((c, i) => (i === index ? { ...c, ...patch } : c)))
  }

  function removeComponent(index) {
    onChange(components.filter((_, i) => i !== index))
  }

  function addComponent() {
    onChange([...components, makeComponent(`Component ${components.length + 1}`)])
  }

  return (
    <div>
      <div className="mb-6 flex items-end gap-6 rounded-lg border border-slate-700 bg-slate-800/50 p-4">
        <div>
          <label className="block text-xs font-medium text-slate-400">Board width (mm)</label>
          <input
            type="number"
            className="mt-1 w-28 rounded bg-slate-800 border border-slate-600 px-2 py-1 text-slate-100"
            value={board.width_mm}
            onChange={(e) => onBoardChange({ ...board, width_mm: Number(e.target.value) })}
          />
        </div>
        <div>
          <label className="block text-xs font-medium text-slate-400">Board height (mm)</label>
          <input
            type="number"
            className="mt-1 w-28 rounded bg-slate-800 border border-slate-600 px-2 py-1 text-slate-100"
            value={board.height_mm}
            onChange={(e) => onBoardChange({ ...board, height_mm: Number(e.target.value) })}
          />
        </div>
      </div>

      <div className="space-y-4">
        {components.map((component, index) => (
          <div key={component.id} className="rounded-lg border border-slate-700 bg-slate-800/50 p-4">
            <div className="flex items-start gap-4">
              <div className="flex-1">
                <label className="block text-xs font-medium text-slate-400">Label / Name</label>
                <input
                  className="mt-1 w-full rounded bg-slate-800 border border-slate-600 px-2 py-1 text-slate-100"
                  placeholder="e.g. FlightController, GPS_Module, ESC_4in1"
                  value={component.label}
                  onChange={(e) => updateComponent(index, { label: e.target.value })}
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-400">Width (mm)</label>
                <input
                  type="number"
                  className="mt-1 w-24 rounded bg-slate-800 border border-slate-600 px-2 py-1 text-slate-100"
                  value={component.width_mm}
                  onChange={(e) => updateComponent(index, { width_mm: Number(e.target.value) })}
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-400">Height (mm)</label>
                <input
                  type="number"
                  className="mt-1 w-24 rounded bg-slate-800 border border-slate-600 px-2 py-1 text-slate-100"
                  value={component.height_mm}
                  onChange={(e) => updateComponent(index, { height_mm: Number(e.target.value) })}
                />
              </div>
              <button
                type="button"
                onClick={() => removeComponent(index)}
                className="mt-5 rounded border border-red-500/40 px-2 py-1 text-sm text-red-400 hover:bg-red-500/10"
              >
                Remove
              </button>
            </div>

            <PinTable pins={component.pins} onChange={(pins) => updateComponent(index, { pins })} />
          </div>
        ))}
      </div>

      <button
        type="button"
        onClick={addComponent}
        className="mt-4 rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-4 py-2 text-emerald-300 hover:bg-emerald-500/20"
      >
        + Add component
      </button>
    </div>
  )
}
