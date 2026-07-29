// Mirrors webapp/backend/tests/test_drone_example.py's fixture, for the
// "Load drone example" convenience button (and used as the fixture for
// this app's own end-to-end browser check).

function pin(number, label, fn = 'signal') {
  return { number, label, function: fn }
}

export function droneExampleProject() {
  const fc = {
    id: 'FC',
    label: 'Flight Controller',
    width_mm: 30,
    height_mm: 30,
    pins: [
      pin('1', 'VCC', 'power'),
      pin('2', 'GND', 'ground'),
      pin('3', 'TX1'),
      pin('4', 'RX1'),
      pin('5', 'PWM1'),
      pin('6', 'PWM2'),
      pin('7', 'PWM3'),
      pin('8', 'PWM4'),
    ],
  }
  const pdb = {
    id: 'PDB',
    label: 'Power Distribution Board',
    width_mm: 25,
    height_mm: 25,
    pins: [
      pin('1', 'VBAT+', 'power'),
      pin('2', 'VBAT-', 'ground'),
      pin('3', '5V_OUT', 'power'),
      pin('4', 'GND_OUT', 'ground'),
    ],
  }
  const gps = {
    id: 'GPS',
    label: 'GPS Module',
    width_mm: 18,
    height_mm: 18,
    pins: [pin('1', 'VCC', 'power'), pin('2', 'GND', 'ground'), pin('3', 'TX'), pin('4', 'RX')],
  }
  const batt = {
    id: 'BATT',
    label: 'Battery Connector',
    width_mm: 8,
    height_mm: 5,
    pins: [pin('1', 'VBAT+', 'power'), pin('2', 'VBAT-', 'ground')],
  }
  const escs = [1, 2, 3, 4].map((i) => ({
    id: `ESC${i}`,
    label: `ESC ${i}`,
    width_mm: 12,
    height_mm: 12,
    pins: [pin('1', 'VCC', 'power'), pin('2', 'GND', 'ground'), pin('3', 'SIGNAL')],
  }))

  const components = [fc, pdb, gps, batt, ...escs]

  const conn = (sc, sp, tc, tp) => ({
    source: { component_id: sc, pin_number: sp },
    target: { component_id: tc, pin_number: tp },
  })

  const connections = [
    conn('BATT', '1', 'PDB', '1'),
    conn('BATT', '2', 'PDB', '2'),
    conn('PDB', '3', 'FC', '1'),
    conn('PDB', '4', 'FC', '2'),
    conn('PDB', '3', 'GPS', '1'),
    conn('PDB', '4', 'GPS', '2'),
    ...escs.flatMap((esc) => [conn('PDB', '3', esc.id, '1'), conn('PDB', '4', esc.id, '2')]),
    ...escs.map((esc, i) => conn('FC', String(i + 5), esc.id, '3')),
    conn('FC', '3', 'GPS', '4'),
    conn('FC', '4', 'GPS', '3'),
  ]

  return { board: { width_mm: 80, height_mm: 80 }, components, connections }
}
