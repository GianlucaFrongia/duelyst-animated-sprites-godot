import { useEffect, useRef, useState } from 'react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'

type Model = {
  name: string; vocab: number; classes: number; dim: number; layers: number; heads: number; mlp: number
  params: number; updated: string; run: string | null; data: string | null; train: Record<string, string>
}
type Dataset = { name: string; pairs: number; colors: number; classes: string[]; size: number; k: number }
type Run = {
  name: string; status: string; pipeline: string; flags: Record<string, string>; metrics: number[][]
  total: number; seconds: number; best: number | null; created: string; params: number; tail: string
}

async function call<T>(url: string, body?: unknown): Promise<T> {
  const r = await fetch(url, body ? { method: 'POST', body: JSON.stringify(body) } : undefined)
  const j = await r.json()
  if (!r.ok) throw new Error(j.error)
  return j
}

const fmt = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(0)}k` : String(n))
const METRICS = ['loss', 'grad norm', 'lr', 'pairs/s'] // columns 1.. of Run.metrics, column 0 is the step
const dur = (s: number) => (s >= 3600 ? `${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m` : `${Math.round(s / 60)}m`)
const COLORS = ['#60a5fa', '#f472b6', '#34d399', '#fbbf24', '#a78bfa', '#f87171']

export default function App() {
  const [models, setModels] = useState<Model[]>([])
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [runs, setRuns] = useState<Run[]>([])
  const refresh = () => {
    call<Model[]>('/api/models').then(setModels)
    call<Dataset[]>('/api/datasets').then(setDatasets)
    call<Run[]>('/api/runs').then(setRuns)
  }
  useEffect(refresh, [])
  const running = runs.some((r) => r.status === 'running')
  useEffect(() => {
    if (!running) return
    const t = setInterval(refresh, 3000)
    return () => clearInterval(t)
  }, [running])

  return (
    <div className="mx-auto max-w-6xl p-4 md:p-8">
      <h1 className="mb-1 text-2xl font-semibold">Sprite Lab</h1>
      <p className="mb-6 text-sm text-muted-foreground">pixel-sprite-gen · models, experiments and sampling</p>
      <Tabs defaultValue="models">
        <TabsList>
          <TabsTrigger value="models">Models</TabsTrigger>
          <TabsTrigger value="datasets">Datasets</TabsTrigger>
          <TabsTrigger value="pipeline">Pipeline</TabsTrigger>
          <TabsTrigger value="inference">Inference</TabsTrigger>
        </TabsList>
        <TabsContent value="models"><ModelsTab models={models} /></TabsContent>
        <TabsContent value="datasets"><DatasetsTab datasets={datasets} /></TabsContent>
        <TabsContent value="pipeline">
          <PipelineTab datasets={datasets} runs={runs} models={models} onChange={refresh} />
        </TabsContent>
        <TabsContent value="inference"><InferenceTab models={models} datasets={datasets} /></TabsContent>
      </Tabs>
    </div>
  )
}

function Pick({ label, value, onChange, items }: { label: string; value: string; onChange: (v: string) => void; items: string[] }) {
  return (
    <div className="grid gap-1.5">
      <Label>{label}</Label>
      <Select value={value} onValueChange={(v) => onChange(String(v ?? ''))}>
        <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
        <SelectContent>{items.map((i) => <SelectItem key={i} value={i}>{i}</SelectItem>)}</SelectContent>
      </Select>
    </div>
  )
}

function Field({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  return (
    <div className="grid gap-1.5">
      <Label>{label}</Label>
      <Input value={value} onChange={(e) => onChange(e.target.value)} />
    </div>
  )
}

// ---------------------------------------------------------------- models

function ModelsTab({ models }: { models: Model[] }) {
  const [sel, setSel] = useState<string>()
  const m = models.find((m) => m.name === sel) ?? models[0]
  return (
    <div className="grid gap-4 md:grid-cols-[1fr_1.2fr]">
      <div className="grid content-start gap-3">
        {models.map((x) => (
          <Card key={x.name} className={`cursor-pointer ${x === m ? 'ring-2 ring-primary' : ''}`} onClick={() => setSel(x.name)}>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                {x.name} <Badge variant="secondary">{fmt(x.params)} params</Badge>
              </CardTitle>
              <CardDescription>
                D {x.dim} · L {x.layers} · H {x.heads} · MLP {x.mlp} · {x.vocab} colors · {x.classes} classes
              </CardDescription>
            </CardHeader>
            <CardContent className="text-xs text-muted-foreground">
              {x.data ? <>data <b>{x.data}</b> · run <b>{x.run}</b></> : 'no tracked run'} · saved {x.updated.slice(0, 16).replace('T', ' ')}
              {Object.keys(x.train).length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1">
                  {Object.entries(x.train).filter(([k]) => k !== 'out' && k !== 'data').map(([k, v]) => (
                    <Badge key={k} variant="outline">{k} {v}</Badge>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>
      {m && <Diagram m={m} />}
    </div>
  )
}

function Box({ title, sub, tone = 'bg-muted' }: { title: string; sub?: string; tone?: string }) {
  return (
    <div className={`rounded-lg border px-3 py-2 text-center ${tone}`}>
      <div className="text-sm font-medium">{title}</div>
      {sub && <div className="text-xs text-muted-foreground">{sub}</div>}
    </div>
  )
}
const Arrow = () => <div className="text-center text-muted-foreground">↓</div>

function Diagram({ m }: { m: Model }) {
  return (
    <Card className="h-fit md:sticky md:top-4">
      <CardHeader>
        <CardTitle>How {m.name} works</CardTitle>
        <CardDescription>MaskGIT-style masked transformer, one frame at a time</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-1.5">
        <div className="grid grid-cols-3 gap-2">
          <Box title="Previous frame" sub="64×64 context, never masked" />
          <Box title="Target frame" sub="64×64, cosine-masked" />
          <Box title="Class" sub={`${m.classes} classes, 10% dropped`} />
        </div>
        <Arrow />
        <Box title="1,024 tokens (2×2 patches)" sub={`4 target + 4 context sub-pixel emb + pos + class → ${m.dim}-d`} />
        <Arrow />
        <div className="rounded-lg border-2 border-dashed p-2">
          <div className="mb-1.5 text-xs text-muted-foreground">× {m.layers} transformer blocks</div>
          <div className="grid grid-cols-2 gap-2">
            <Box title="LayerNorm → Attention" sub={`${m.heads} heads × ${m.dim / m.heads}-d, bidirectional`} tone="bg-blue-500/10" />
            <Box title="LayerNorm → MLP" sub={`${m.dim} → ${m.mlp} → ${m.dim}, GELU`} tone="bg-pink-500/10" />
          </div>
        </div>
        <Arrow />
        <Box title="4 pixel heads per token" sub={`softmax over ${m.vocab} palette colors · loss on masked tokens only`} />
        <Arrow />
        <Box title="Sampling loop" sub="~12 steps: keep most confident pixels, re-mask the rest → frame becomes next context" tone="bg-emerald-500/10" />
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------- datasets

type Page = { total: number; counts: number[]; palette: number[][]; items: { i: number; label: number; px: string }[] }
const PAGE = 48

// K frames stacked vertically in `px` are drawn side by side
function Example({ px, palette, size, k }: { px: string; palette: number[][]; size: number; k: number }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current!.getContext('2d')!
    const img = c.createImageData(size * k, size)
    const bytes = atob(px)
    for (let j = 0; j < bytes.length; j++) {
      const x = j % size, y = Math.floor(j / size), f = Math.floor(y / size)
      const o = ((y % size) * size * k + f * size + x) * 4, col = palette[bytes.charCodeAt(j)] ?? [255, 0, 255]
      img.data.set([col[0], col[1], col[2], bytes.charCodeAt(j) === 0 ? 0 : 255], o)
    }
    c.putImageData(img, 0, 0)
  }, [px, palette, size, k])
  return <canvas ref={ref} width={size * k} height={size} className="w-full rounded border bg-muted [image-rendering:pixelated]" />
}

function DatasetsTab({ datasets }: { datasets: Dataset[] }) {
  const [name, setName] = useState('')
  const [cls, setCls] = useState<number>()
  const [offset, setOffset] = useState(0)
  const [page, setPage] = useState<Page>()
  const [err, setErr] = useState('')
  const d = datasets.find((d) => d.name === name) ?? datasets[0]
  useEffect(() => {
    if (!d) return
    setErr('')
    call<Page>('/api/examples', { data: d.name, cls, offset, limit: PAGE }).then(setPage, (e) => setErr(e.message))
  }, [d?.name, cls, offset])
  const pick = (f: () => void) => { f(); setOffset(0) }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Dataset explorer</CardTitle>
        <CardDescription>
          {d && <>{d.pairs.toLocaleString()} examples · {d.k} frames of {d.size}×{d.size} each · {d.colors} colors (index 0 is transparent)</>}
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4">
        <div className="w-64"><Pick label="Dataset" value={d?.name ?? ''} onChange={(v) => pick(() => { setName(v); setCls(undefined); setPage(undefined) })} items={datasets.map((x) => x.name)} /></div>
        <div className="flex flex-wrap gap-2">
          <Button size="sm" variant={cls === undefined ? 'default' : 'outline'} onClick={() => pick(() => setCls(undefined))}>all</Button>
          {d?.classes.map((c, i) => (
            <Button key={c} size="sm" variant={cls === i ? 'default' : 'outline'} onClick={() => pick(() => setCls(i))}>
              {c} <span className="ml-1 opacity-60">{page?.counts[i]?.toLocaleString()}</span>
            </Button>
          ))}
        </div>
        {err && <pre className="text-xs text-destructive">{err}</pre>}
        {page && d && (
          <>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
              {page.items.map((e) => (
                <div key={e.i} title={`example ${e.i}`}>
                  <Example px={e.px} palette={page.palette} size={d.size} k={d.k} />
                  <div className="mt-1 flex justify-between text-xs text-muted-foreground"><span>#{e.i}</span><span>{d.classes[e.label]}</span></div>
                </div>
              ))}
            </div>
            <div className="flex items-center gap-3 text-sm">
              <Button size="sm" variant="outline" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>Prev</Button>
              <span className="text-muted-foreground">{page.total ? offset + 1 : 0}–{Math.min(offset + PAGE, page.total)} of {page.total.toLocaleString()}</span>
              <Button size="sm" variant="outline" disabled={offset + PAGE >= page.total} onClick={() => setOffset(offset + PAGE)}>Next</Button>
            </div>
            <div className="flex flex-wrap gap-1">
              {page.palette.map((c, i) => <span key={i} title={`${i}: ${c.join(' ')}`} className="size-5 rounded border" style={{ background: `rgb(${c})` }} />)}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------- pipeline

const DEFAULTS: Record<string, string> = {
  steps: '4000', dim: '64', layers: '4', heads: '4', mlp: '', batch: '8', lr: '1e-3', warmup: '100',
  'cfg-drop': '0.1', 'context-drop': '0.15', flip: '0.0', seed: '0',
}

function PipelineTab({ datasets, runs, models, onChange }: { datasets: Dataset[]; runs: Run[]; models: Model[]; onChange: () => void }) {
  const [name, setName] = useState('')
  const [data, setData] = useState('')
  const [resume, setResume] = useState('none')
  const [flags, setFlags] = useState(DEFAULTS)
  const [msg, setMsg] = useState('')
  const [picked, setPicked] = useState<string[]>([])
  const [metric, setMetric] = useState('loss')
  const ds = data || datasets[0]?.name || ''
  const shown = runs.filter((r) => picked.includes(r.name))

  const start = async () => {
    setMsg('starting…')
    try {
      const r = await call<{ out: string }>('/api/train', { name, data: ds, flags, resume: resume === 'none' ? '' : resume })
      setMsg(r.out); setPicked((p) => [...p, name]); onChange()
    } catch (e) { setMsg((e as Error).message) }
  }
  const stop = (n: string) => call('/api/stop', { name: n }).then(onChange, (e) => setMsg(e.message))
  const reuse = (r: Run) => {
    setFlags({ ...DEFAULTS, ...Object.fromEntries(Object.entries(r.flags).filter(([k]) => k in DEFAULTS)) })
    if (r.flags.data) setData(r.flags.data)
    setName(`${r.name}-2`)
  }

  return (
    <div className="grid gap-4">
      <Card>
        <CardHeader>
          <CardTitle>Train a model</CardTitle>
          <CardDescription>Runs <code>./psg train &lt;name&gt; -- …</code>; the checkpoint goes to checkpoints/&lt;name&gt;, the run to runs/&lt;name&gt;.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4">
          <div className="grid gap-3 sm:grid-cols-3">
            <Field label="Run name" value={name} onChange={setName} />
            <Pick label="Dataset" value={ds} onChange={setData} items={datasets.map((d) => d.name)} />
            <Pick label="Resume from" value={resume} onChange={setResume} items={['none', ...models.map((m) => m.name)]} />
          </div>
          {datasets.find((d) => d.name === ds) && (() => {
            const d = datasets.find((d) => d.name === ds)!
            return <p className="text-xs text-muted-foreground">{d.pairs.toLocaleString()} frame pairs · {d.colors} colors · classes: {d.classes.join(', ')}</p>
          })()}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
            {Object.keys(DEFAULTS).map((k) => (
              <Field key={k} label={k === 'mlp' ? 'mlp (4×dim)' : k} value={flags[k]} onChange={(v) => setFlags({ ...flags, [k]: v })} />
            ))}
          </div>
          <div className="flex items-center gap-3">
            <Button onClick={start} disabled={!/^[\w.-]+$/.test(name) || !ds}>Start training</Button>
            <pre className="truncate text-xs text-muted-foreground">{msg}</pre>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Experiments</CardTitle>
          <CardDescription>Every run is kept in runs/; its metrics are saved to runs/&lt;name&gt;/metrics.csv and summary.json. Tick runs to compare them.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4">
          {shown.length > 0 && (
            <div className="grid gap-2">
              <div className="w-40"><Pick label="Metric" value={metric} onChange={setMetric} items={METRICS} /></div>
              <MetricChart runs={shown} col={METRICS.indexOf(metric) + 1} />
            </div>
          )}
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead /><TableHead>Run</TableHead><TableHead>Status</TableHead><TableHead>Data</TableHead>
                <TableHead>Hyperparameters</TableHead><TableHead>Progress</TableHead><TableHead>Loss</TableHead><TableHead>Best</TableHead><TableHead>Time</TableHead><TableHead />
              </TableRow>
            </TableHeader>
            <TableBody>
              {runs.map((r) => {
                const last = r.metrics.at(-1)
                const i = picked.indexOf(r.name)
                return (
                  <TableRow key={r.name}>
                    <TableCell>
                      <input type="checkbox" aria-label={`compare ${r.name}`} checked={i >= 0} style={{ accentColor: COLORS[i % COLORS.length] }}
                        onChange={() => setPicked(i >= 0 ? picked.filter((p) => p !== r.name) : [...picked, r.name])} />
                    </TableCell>
                    <TableCell className="font-medium">{r.name}<div className="text-xs text-muted-foreground">{r.created}</div></TableCell>
                    <TableCell><Badge variant={r.status === 'failed' ? 'destructive' : r.status === 'running' ? 'default' : 'secondary'}>{r.status}</Badge></TableCell>
                    <TableCell>{r.flags.data ?? '—'}</TableCell>
                    <TableCell className="max-w-72 whitespace-normal text-xs">
                      {Object.entries(r.flags).filter(([k]) => !['data', 'out', 'log-every', 'save-every'].includes(k)).map(([k, v]) => `${k} ${v}`).join(' · ')}
                    </TableCell>
                    <TableCell className="text-xs">{last ? `${last[0]} / ${r.total}` : '—'}</TableCell>
                    <TableCell className="text-xs">{last ? last[1].toFixed(4) : '—'}</TableCell>
                    <TableCell className="text-xs">{r.best?.toFixed(4) ?? '—'}</TableCell>
                    <TableCell className="text-xs">{r.seconds ? dur(r.seconds) : '—'}</TableCell>
                    <TableCell className="space-x-1 text-right">
                      <Button size="sm" variant="outline" onClick={() => reuse(r)}>Reuse</Button>
                      {r.status === 'running' && <Button size="sm" variant="destructive" onClick={() => stop(r.name)}>Stop</Button>}
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
          {shown.map((r) => (
            <details key={r.name} className="text-xs">
              <summary className="cursor-pointer text-muted-foreground">{r.name}: pipeline and log tail</summary>
              <pre className="mt-2 overflow-x-auto rounded bg-muted p-2">{r.pipeline}{'\n'}{r.tail}</pre>
            </details>
          ))}
        </CardContent>
      </Card>
    </div>
  )
}

function MetricChart({ runs, col }: { runs: Run[]; col: number }) {
  const W = 800, H = 220, P = 32
  const pts = runs.flatMap((r) => r.metrics)
  if (!pts.length) return <p className="text-sm text-muted-foreground">No metrics logged yet.</p>
  const maxX = Math.max(...pts.map((p) => p[0]))
  const ys = pts.map((p) => p[col]).sort((a, b) => a - b)
  const minY = ys[0], maxY = ys[Math.floor(ys.length * 0.98)] // clip the warmup spike
  const x = (v: number) => P + (v / maxX) * (W - 2 * P)
  const y = (v: number) => H - P - ((Math.min(v, maxY) - minY) / (maxY - minY || 1)) * (H - 2 * P)
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={`${METRICS[col - 1]} per step`}>
        {[minY, (minY + maxY) / 2, maxY].map((v) => (
          <g key={v}>
            <line x1={P} x2={W - P} y1={y(v)} y2={y(v)} stroke="currentColor" opacity={0.1} />
            <text x={2} y={y(v) + 4} fontSize={10} fill="currentColor" opacity={0.6}>{v < 0.01 ? v.toExponential(1) : v.toFixed(2)}</text>
          </g>
        ))}
        <text x={W - P} y={H - 8} fontSize={10} textAnchor="end" fill="currentColor" opacity={0.6}>step {maxX}</text>
        {runs.map((r, i) => (
          <polyline key={r.name} fill="none" strokeWidth={1.5} stroke={COLORS[i % COLORS.length]}
            points={r.metrics.map((m) => `${x(m[0])},${y(m[col])}`).join(' ')} />
        ))}
      </svg>
      <div className="flex flex-wrap gap-3 text-xs">
        {runs.map((r, i) => <span key={r.name}><span style={{ color: COLORS[i % COLORS.length] }}>■</span> {r.name}</span>)}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------- inference

function InferenceTab({ models, datasets }: { models: Model[]; datasets: Dataset[] }) {
  const [model, setModel] = useState('')
  const [data, setData] = useState('')
  const [cls, setCls] = useState('')
  const [opts, setOpts] = useState({ n: '4', frames: '8', steps: '12', temperature: '0.7', guidance: '2.0', seed: '0', init: '' })
  const [busy, setBusy] = useState(false)
  const [res, setRes] = useState<{ gifs: string[]; grid: string | null; log: string; dir: string } | null>(null)
  const [err, setErr] = useState('')

  const m = models.find((x) => x.name === model) ?? models[0]
  // a checkpoint only works with datasets that have the same palette size and class count
  const fits = datasets.filter((d) => m && d.colors === m.vocab && d.classes.length === m.classes)
  const ds = fits.find((d) => d.name === data) ?? fits.find((d) => d.name === m?.data) ?? fits[0]
  const c = ds?.classes.includes(cls) ? cls : ds?.classes[0] ?? ''

  const go = async () => {
    setBusy(true); setErr('')
    try { setRes(await call('/api/sample', { model: m.name, data: ds.name, class: c, ...opts })) } catch (e) { setErr((e as Error).message) }
    setBusy(false)
  }

  return (
    <div className="grid gap-4 md:grid-cols-[320px_1fr]">
      <Card className="h-fit">
        <CardHeader>
          <CardTitle>Prompt</CardTitle>
          <CardDescription>The model is class-conditioned: the class is the prompt.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3">
          <Pick label="Model" value={m?.name ?? ''} onChange={setModel} items={models.map((x) => x.name)} />
          <Pick label="Dataset (palette)" value={ds?.name ?? ''} onChange={setData} items={fits.map((d) => d.name)} />
          <Pick label="Class" value={c} onChange={setCls} items={[...(ds?.classes ?? []), 'none']} />
          <div className="grid grid-cols-2 gap-3">
            {Object.entries(opts).map(([k, v]) => (
              <Field key={k} label={k === 'init' ? 'init pair (opt.)' : k} value={v} onChange={(nv) => setOpts({ ...opts, [k]: nv })} />
            ))}
          </div>
          <Button onClick={go} disabled={busy || !m || !ds}>{busy ? 'Sampling…' : 'Generate'}</Button>
          {err && <pre className="whitespace-pre-wrap text-xs text-destructive">{err}</pre>}
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Output</CardTitle>
          <CardDescription>{res ? `saved in ${res.dir}` : 'Generated animations show up here.'}</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4">
          {res && (
            <>
              <div className="flex flex-wrap gap-3">
                {res.gifs.map((g) => <img key={g} src={g} alt="generated animation" className="w-40 rounded border [image-rendering:pixelated]" />)}
              </div>
              {res.grid && <img src={res.grid} alt="all frames" className="w-full rounded border [image-rendering:pixelated]" />}
              <details className="text-xs"><summary className="cursor-pointer text-muted-foreground">sample log</summary><pre className="mt-2 overflow-x-auto">{res.log}</pre></details>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
