const FUNCTIONS = ['power', 'ground', 'signal', 'high_speed']

const FUNCTION_COLORS = {
  power: 'bg-red-500/20 text-red-300 border-red-500/40',
  ground: 'bg-slate-500/20 text-slate-300 border-slate-500/40',
  signal: 'bg-sky-500/20 text-sky-300 border-sky-500/40',
  high_speed: 'bg-amber-500/20 text-amber-300 border-amber-500/40',
}

export default function PinTable({ pins, onChange }) {
  function updatePin(index, patch) {
    onChange(pins.map((p, i) => (i === index ? { ...p, ...patch } : p)))
  }

  function removePin(index) {
    onChange(pins.filter((_, i) => i !== index))
  }

  function addPin() {
    const nextNumber = String(pins.length + 1)
    onChange([...pins, { number: nextNumber, label: '', function: 'signal' }])
  }

  return (
    <div className="mt-3">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-slate-400">
            <th className="w-20 pb-1 font-medium">Pin #</th>
            <th className="pb-1 font-medium">Label</th>
            <th className="w-40 pb-1 font-medium">Function</th>
            <th className="w-10 pb-1" />
          </tr>
        </thead>
        <tbody>
          {pins.map((pin, index) => (
            <tr key={index} className="border-t border-slate-700/60">
              <td className="py-1 pr-2">
                <input
                  className="w-full rounded bg-slate-800 border border-slate-600 px-2 py-1 text-slate-100"
                  value={pin.number}
                  onChange={(e) => updatePin(index, { number: e.target.value })}
                />
              </td>
              <td className="py-1 pr-2">
                <input
                  className="w-full rounded bg-slate-800 border border-slate-600 px-2 py-1 text-slate-100"
                  placeholder="e.g. VCC, TX1, PWM1"
                  value={pin.label}
                  onChange={(e) => updatePin(index, { label: e.target.value })}
                />
              </td>
              <td className="py-1 pr-2">
                <select
                  className={`w-full rounded border px-2 py-1 ${FUNCTION_COLORS[pin.function] ?? ''}`}
                  value={pin.function}
                  onChange={(e) => updatePin(index, { function: e.target.value })}
                >
                  {FUNCTIONS.map((f) => (
                    <option key={f} value={f} className="bg-slate-800 text-slate-100">
                      {f}
                    </option>
                  ))}
                </select>
              </td>
              <td className="py-1 text-center">
                <button
                  type="button"
                  onClick={() => removePin(index)}
                  className="text-slate-500 hover:text-red-400"
                  title="Remove pin"
                >
                  ✕
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <button
        type="button"
        onClick={addPin}
        className="mt-2 text-sm text-emerald-400 hover:text-emerald-300"
      >
        + Add pin
      </button>
    </div>
  )
}
