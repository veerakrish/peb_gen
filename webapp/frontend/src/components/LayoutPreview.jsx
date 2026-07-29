import { useEffect, useRef, useState } from 'react'
import { kicadDownloadUrl, gerberDownloadUrl } from '../api'

const LAYER_COLORS = { 'F.Cu': '#ef4444', 'B.Cu': '#3b82f6' }

// Mirrors webapp/backend/app/footprint.py's assign_pin_local_positions /
// to_global exactly, so pads line up with the traces the API returned.
function assignPinLocalPositions(component) {
  const { width_mm: w, height_mm: h, pins } = component
  const n = pins.length
  if (n === 0) return {}
  const perimeter = 2 * (w + h)
  const positions = {}
  pins.forEach((pin, i) => {
    const t = ((i + 0.5) / n) * perimeter
    positions[pin.number] = perimeterPoint(t, w, h)
  })
  return positions
}

function perimeterPoint(t, w, h) {
  if (t < w) return [-w / 2 + t, -h / 2]
  t -= w
  if (t < h) return [w / 2, -h / 2 + t]
  t -= h
  if (t < w) return [w / 2 - t, h / 2]
  t -= w
  return [-w / 2, h / 2 - t]
}

function toGlobal([lx, ly], x, y, rotationDeg) {
  const theta = (rotationDeg * Math.PI) / 180
  const cos = Math.cos(theta)
  const sin = Math.sin(theta)
  return [x + lx * cos - ly * sin, y + lx * sin + ly * cos]
}

export default function LayoutPreview({ project, result, sessionId }) {
  const canvasRef = useRef(null)
  const containerRef = useRef(null)
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const dragState = useRef(null)

  const byId = Object.fromEntries(project.components.map((c) => [c.id, c]))
  const placementById = Object.fromEntries(result.placement.map((p) => [p.component_id, p]))

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const cssWidth = canvas.clientWidth
    const cssHeight = canvas.clientHeight
    canvas.width = cssWidth * devicePixelRatio
    canvas.height = cssHeight * devicePixelRatio
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0)

    const fitScale = Math.min(cssWidth / result.board_width_mm, cssHeight / result.board_height_mm) * 0.9
    const scale = fitScale * zoom
    const originX = (cssWidth - result.board_width_mm * scale) / 2 + pan.x
    const originY = (cssHeight - result.board_height_mm * scale) / 2 + pan.y
    const toPx = (xMm, yMm) => [originX + xMm * scale, originY + yMm * scale]

    ctx.fillStyle = '#0f172a'
    ctx.fillRect(0, 0, cssWidth, cssHeight)

    // board outline, dark solder-mask green
    const [bx0, by0] = toPx(0, 0)
    ctx.fillStyle = '#0a3d1f'
    ctx.fillRect(bx0, by0, result.board_width_mm * scale, result.board_height_mm * scale)
    ctx.strokeStyle = '#fbbf24'
    ctx.lineWidth = 1.5
    ctx.strokeRect(bx0, by0, result.board_width_mm * scale, result.board_height_mm * scale)

    // copper traces
    for (const seg of result.segments) {
      ctx.strokeStyle = LAYER_COLORS[seg.layer] ?? '#aaa'
      ctx.lineWidth = Math.max(1, 0.25 * scale)
      ctx.lineJoin = 'round'
      ctx.lineCap = 'round'
      ctx.beginPath()
      seg.points.forEach(([x, y], i) => {
        const [px, py] = toPx(x, y)
        if (i === 0) ctx.moveTo(px, py)
        else ctx.lineTo(px, py)
      })
      ctx.stroke()
    }

    // vias
    for (const via of result.vias) {
      const [px, py] = toPx(via.position[0], via.position[1])
      ctx.beginPath()
      ctx.arc(px, py, Math.max(2, 0.3 * scale), 0, 2 * Math.PI)
      ctx.fillStyle = '#facc15'
      ctx.fill()
      ctx.strokeStyle = '#78350f'
      ctx.lineWidth = 1
      ctx.stroke()
    }

    // components: silkscreen outline + label + pads
    for (const placement of result.placement) {
      const component = byId[placement.component_id]
      if (!component) continue
      const [px, py] = toPx(placement.x_mm, placement.y_mm)

      ctx.save()
      ctx.translate(px, py)
      ctx.rotate((placement.rotation_deg * Math.PI) / 180)
      ctx.strokeStyle = '#e2e8f0'
      ctx.lineWidth = 1
      ctx.strokeRect(
        (-component.width_mm / 2) * scale,
        (-component.height_mm / 2) * scale,
        component.width_mm * scale,
        component.height_mm * scale
      )
      ctx.fillStyle = '#e2e8f0'
      ctx.font = `${Math.max(9, 3.2 * scale)}px sans-serif`
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(component.label, 0, 0)
      ctx.restore()

      const localPositions = assignPinLocalPositions(component)
      for (const pin of component.pins) {
        const [gx, gy] = toGlobal(localPositions[pin.number], placement.x_mm, placement.y_mm, placement.rotation_deg)
        const [pxPad, pyPad] = toPx(gx, gy)
        ctx.beginPath()
        ctx.arc(pxPad, pyPad, Math.max(1.5, 0.2 * scale), 0, 2 * Math.PI)
        ctx.fillStyle = '#d4af37'
        ctx.fill()
      }
    }
  }, [result, project, zoom, pan, byId])

  function onWheel(e) {
    e.preventDefault()
    const delta = -e.deltaY * 0.001
    setZoom((z) => Math.min(8, Math.max(0.2, z * (1 + delta))))
  }

  function onMouseDown(e) {
    dragState.current = { startX: e.clientX, startY: e.clientY, panX: pan.x, panY: pan.y }
  }
  function onMouseMove(e) {
    if (!dragState.current) return
    const dx = e.clientX - dragState.current.startX
    const dy = e.clientY - dragState.current.startY
    setPan({ x: dragState.current.panX + dx, y: dragState.current.panY + dy })
  }
  function onMouseUp() {
    dragState.current = null
  }

  const metrics = result.metrics

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_280px]">
      <div>
        <div
          ref={containerRef}
          className="h-[520px] w-full overflow-hidden rounded-lg border border-slate-700"
        >
          <canvas
            ref={canvasRef}
            className="h-full w-full cursor-grab active:cursor-grabbing"
            onWheel={onWheel}
            onMouseDown={onMouseDown}
            onMouseMove={onMouseMove}
            onMouseUp={onMouseUp}
            onMouseLeave={onMouseUp}
          />
        </div>
        <div className="mt-2 flex items-center gap-2 text-sm">
          <button
            type="button"
            onClick={() => setZoom((z) => Math.min(8, z * 1.25))}
            className="rounded border border-slate-600 px-2 py-1 text-slate-300 hover:bg-slate-700"
          >
            Zoom +
          </button>
          <button
            type="button"
            onClick={() => setZoom((z) => Math.max(0.2, z / 1.25))}
            className="rounded border border-slate-600 px-2 py-1 text-slate-300 hover:bg-slate-700"
          >
            Zoom −
          </button>
          <button
            type="button"
            onClick={() => {
              setZoom(1)
              setPan({ x: 0, y: 0 })
            }}
            className="rounded border border-slate-600 px-2 py-1 text-slate-300 hover:bg-slate-700"
          >
            Reset view
          </button>
          <span className="ml-2 text-slate-500">Scroll to zoom, drag to pan</span>
          <span className="ml-auto flex items-center gap-3 text-slate-400">
            <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-red-500" />F.Cu</span>
            <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-blue-500" />B.Cu</span>
            <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-yellow-400" />Via</span>
          </span>
        </div>
      </div>

      <div className="space-y-4">
        <div className="rounded-lg border border-slate-700 bg-slate-800/50 p-4">
          <h3 className="mb-2 text-sm font-semibold text-slate-300">Metrics</h3>
          <dl className="space-y-1.5 text-sm">
            <Stat label="Attempts tried" value={metrics.attempts_tried} />
            <Stat
              label="Unrouted nets"
              value={metrics.unrouted_nets_count}
              warn={metrics.unrouted_nets_count > 0}
            />
            <Stat label="Total trace length" value={`${metrics.total_trace_length_mm.toFixed(1)} mm`} />
            <Stat label="Vias" value={metrics.via_count} />
            <Stat label="Overlap area" value={`${metrics.overlap_area_mm2.toFixed(2)} mm²`} />
            <Stat label="Out-of-bounds area" value={`${metrics.out_of_bounds_area_mm2.toFixed(2)} mm²`} />
          </dl>
          {metrics.unrouted_nets_count > 0 && (
            <p className="mt-2 text-xs text-amber-400">
              Unrouted: {metrics.unrouted_nets.join(', ')}
            </p>
          )}
        </div>

        <div className="rounded-lg border border-slate-700 bg-slate-800/50 p-4">
          <h3 className="mb-2 text-sm font-semibold text-slate-300">Export</h3>
          <div className="flex flex-col gap-2">
            <a
              href={kicadDownloadUrl(sessionId)}
              className="rounded-lg bg-emerald-500 px-4 py-2 text-center font-medium text-slate-900 hover:bg-emerald-400"
            >
              Download .kicad_pcb
            </a>
            <a
              href={gerberDownloadUrl(sessionId)}
              className="rounded-lg border border-slate-600 px-4 py-2 text-center font-medium text-slate-200 hover:bg-slate-700"
            >
              Download Gerbers (.zip)
            </a>
          </div>
        </div>
      </div>
    </div>
  )
}

function Stat({ label, value, warn }) {
  return (
    <div className="flex justify-between">
      <dt className="text-slate-400">{label}</dt>
      <dd className={warn ? 'font-medium text-amber-400' : 'font-medium text-slate-100'}>{value}</dd>
    </div>
  )
}
