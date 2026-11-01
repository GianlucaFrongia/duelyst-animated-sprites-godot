// Dev-server API over the project folder: reads checkpoints/, data*/, runs/ and calls psg, sample-<arch> and render.py.
import { execFile } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import type { IncomingMessage, ServerResponse } from 'node:http'
import type { Plugin } from 'vite'

const ROOT = path.resolve(import.meta.dirname, '..')
const p = (...xs: string[]) => path.join(ROOT, ...xs)
const read = (f: string) => (fs.existsSync(f) ? fs.readFileSync(f, 'utf8') : '')
const kv = (text: string) =>
  Object.fromEntries(text.split('\n').filter(Boolean).map((l) => l.trim().split(/\s+/, 2)))
const NAME = /^[\w.-]+$/
const TRAIN_FLAGS = ['steps', 'dim', 'layers', 'heads', 'mlp', 'batch', 'lr', 'warmup', 'weight-decay',
  'cfg-drop', 'context-drop', 'flip', 'log-every', 'save-every', 'seed']

function run(cmd: string, args: string[]) {
  return new Promise<string>((resolve, reject) =>
    execFile(cmd, args, { cwd: ROOT, maxBuffer: 1 << 24 }, (err, out, errOut) =>
      err ? reject(new Error(errOut || out || err.message)) : resolve(out + errOut)))
}

function alive(pidFile: string) {
  const pid = Number(read(pidFile))
  try { return pid > 0 && process.kill(pid, 0) } catch { return false }
}

// `--key value` pairs from a train_gpu command line; psg quotes values as 'x'.
function cmdFlags(cmd: string) {
  const toks = cmd.replace(/'/g, '').split(/\s+/)
  const out: Record<string, string> = {}
  toks.forEach((t, i) => { if (t.startsWith('--') && toks[i + 1] && !toks[i + 1].startsWith('--')) out[t.slice(2)] = toks[i + 1] })
  return out
}

// Architectures (models/<arch>/arch/variant.mojo); each tool is built per architecture as ./<tool>-<arch>.
const archs = () => fs.readdirSync(p('models')).filter((a) => fs.existsSync(p('models', a, 'arch', 'variant.mojo')))
const ckptArch = (name: string) => kv(read(p('checkpoints', `${name}.cfg`))).A ?? ''

function datasets() {
  return fs.readdirSync(ROOT).filter((d) => fs.existsSync(p(d, 'meta.txt'))).map((name) => {
    const m = kv(read(p(name, 'meta.txt')))
    return { name, pairs: +m.N, colors: +m.V, classes: read(p(name, 'classes.txt')).split('\n').filter(Boolean), size: +m.S, k: +m.K || 2 }
  })
}

function runs() {
  if (!fs.existsSync(p('runs'))) return []
  return fs.readdirSync(p('runs')).filter((n) => fs.existsSync(p('runs', n, 'meta'))).map((name) => {
    const dir = p('runs', name)
    let pipeline = read(path.join(dir, 'pipeline'))
    for (const [, k, v] of pipeline.matchAll(/^set (\w+) (\S+)/gm)) pipeline = pipeline.replaceAll(`\${${k}}`, v)
    const cmds = pipeline.split('\n').filter((l) => l.trim().startsWith('cmd ')).map((l) => l.trim().slice(4))
    const train = cmds.find((c) => c.startsWith('./train_gpu')) ?? ''
    const rcs = fs.readdirSync(dir).filter((f) => f.endsWith('.rc')).map((f) => read(path.join(dir, f)).trim())
    const status = alive(path.join(dir, 'worker.pid')) ? 'running'
      : fs.existsSync(path.join(dir, 'stopped')) ? 'stopped'
      : rcs.some((r) => r !== '0') ? 'failed' : 'done'
    // metrics from every train_gpu log line: `step a / b | loss x | grad norm g | lr l | s pairs/s | ...`
    const logs = fs.readdirSync(dir).filter((f) => /^\d+\.log$/.test(f)).sort((a, b) => parseInt(a) - parseInt(b))
    const metrics: number[][] = []
    let total = 0
    for (const f of logs)
      for (const m of read(path.join(dir, f)).matchAll(
        /step (\d+) \/ (\d+) \| loss ([\d.e+-]+) \| grad norm ([\d.e+-]+) \| lr ([\d.e+-]+) \| ([\d.e+-]+) pairs\/s/g)) {
        metrics.push([+m[1], +m[3], +m[4], +m[5], +m[6]]); total = +m[2]
      }
    const log = logs.length ? read(path.join(dir, logs.at(-1)!)) : ''
    // wall time of every stage that started; a running stage counts up to now
    const seconds = logs.reduce((s, f) => {
      const i = parseInt(f), start = +read(path.join(dir, `${i}.start`))
      return start ? s + (+read(path.join(dir, `${i}.end`)) || Date.now() / 1000) - start : s
    }, 0)
    const losses = metrics.map((r) => r[1])
    const summary = {
      name, status, steps: metrics.at(-1)?.[0] ?? 0, total, seconds: Math.round(seconds),
      final_loss: losses.at(-1) ?? null, best_loss: losses.length ? Math.min(...losses) : null,
      params: +(log.match(/model: (\d+) parameters/)?.[1] ?? 0), flags: cmdFlags(train),
    }
    if (metrics.length) { // saved next to the run so experiments can be compared outside the UI too
      fs.writeFileSync(path.join(dir, 'metrics.csv'), ['step,loss,grad_norm,lr,pairs_per_s', ...metrics.map((r) => r.join(','))].join('\n') + '\n')
      fs.writeFileSync(path.join(dir, 'summary.json'), JSON.stringify(summary, null, 2) + '\n')
    }
    return {
      name, status, pipeline, flags: summary.flags, metrics, total, seconds: summary.seconds, best: summary.best_loss,
      created: kv(read(path.join(dir, 'meta'))).created ?? '',
      createdAt: fs.statSync(path.join(dir, 'meta')).mtimeMs,
      params: summary.params,
      tail: log.split('\n').slice(-12).join('\n'),
    }
  }).sort((a, b) => b.createdAt - a.createdAt)
}

function models() {
  const all = runs()
  return fs.readdirSync(p('checkpoints')).filter((f) => f.endsWith('.cfg')).map((f) => {
    const name = f.slice(0, -4)
    const c = kv(read(p('checkpoints', f)))
    const bin = p('checkpoints', name + '.bin')
    const r = all.find((r) => r.flags.out === `checkpoints/${name}`)
    return {
      name, arch: c.A ?? '', vocab: +c.V, classes: +c.C, dim: +c.D, layers: +c.L, heads: +c.H, mlp: +c.F,
      params: fs.existsSync(bin) ? fs.statSync(bin).size / 4 : 0, // checkpoint = float32 params
      updated: fs.existsSync(bin) ? fs.statSync(bin).mtime.toISOString() : '',
      run: r?.name ?? null, data: r?.flags.data ?? null, train: r?.flags ?? {},
    }
  })
}

async function body(req: IncomingMessage) {
  let s = ''
  for await (const c of req) s += c
  return JSON.parse(s || '{}')
}

const num = (v: unknown) => {
  if (!/^-?[\d.]+(e-?\d+)?$/.test(String(v))) throw new Error(`not a number: ${v}`)
  return String(v)
}

async function api(url: string, req: IncomingMessage) {
  if (url === '/api/models') return models()
  if (url === '/api/datasets') return datasets()
  if (url === '/api/runs') return runs()
  if (url === '/api/archs') return archs()
  if (req.method !== 'POST') throw new Error('not found')
  const b = await body(req)
  if (url === '/api/train') {
    if (!NAME.test(b.name) || !datasets().some((d) => d.name === b.data)) throw new Error('bad name or dataset')
    const args = ['train', b.name, '--detach']
    if (b.arch && !b.resume) { if (!archs().includes(b.arch)) throw new Error('bad architecture'); args.push('--arch', b.arch) }
    args.push('--', '--data', b.data)
    for (const k of TRAIN_FLAGS) if (b.flags?.[k] !== undefined && b.flags[k] !== '') args.push(`--${k}`, num(b.flags[k]))
    if (b.resume) { if (!NAME.test(b.resume)) throw new Error('bad resume'); args.push('--resume', `checkpoints/${b.resume}`) }
    return { out: await run('./psg', args) }
  }
  if (url === '/api/examples') { // one page of examples: K stacked 64x64 frames of palette indices, base64
    const d = datasets().find((d) => d.name === b.data)
    if (!d) throw new Error('bad dataset')
    const labels = fs.readFileSync(p(d.name, 'labels.bin'))
    const ids = [...labels.keys()].filter((i) => b.cls === undefined || labels[i] === b.cls)
    const counts = d.classes.map((_, c) => labels.filter((l) => l === c).length)
    const palette = read(p(d.name, 'palette.txt')).split('\n').filter(Boolean).map((l) => l.trim().split(/\s+/).map(Number))
    const len = d.k * d.size * d.size, fd = fs.openSync(p(d.name, 'pixels.bin'), 'r')
    const items = ids.slice(b.offset, b.offset + Math.min(b.limit, 200)).map((i) => {
      const buf = Buffer.alloc(len)
      fs.readSync(fd, buf, 0, len, i * len)
      return { i, label: labels[i], px: buf.toString('base64') }
    })
    fs.closeSync(fd)
    return { total: ids.length, counts, palette, items }
  }
  if (url === '/api/stop') {
    if (!NAME.test(b.name)) throw new Error('bad name')
    return { out: await run('./psg', ['stop', b.name]) }
  }
  if (url === '/api/sample') {
    if (!NAME.test(b.model) || !datasets().some((d) => d.name === b.data) || !NAME.test(b.class)) throw new Error('bad input')
    const dir = `out/ui/${b.model}-${Date.now()}`
    const args = ['--ckpt', `checkpoints/${b.model}`, '--data', b.data, '--class', b.class, '--out', `${dir}/samples.bin`]
    for (const k of ['n', 'frames', 'steps', 'temperature', 'guidance', 'seed', 'init'])
      if (b[k] !== undefined && b[k] !== '') args.push(`--${k}`, num(b[k]))
    fs.mkdirSync(p(dir), { recursive: true })
    fs.writeFileSync(p(dir, 'request.json'), JSON.stringify(b, null, 2))
    const arch = ckptArch(b.model)
    if (!archs().includes(arch)) throw new Error(`checkpoint ${b.model} has no known architecture`)
    const log = await run(`./sample-${arch}`, args)
    await run('.venv/bin/python', ['tools/render.py', '--samples', `${dir}/samples.bin`, '--palette', `${b.data}/palette.txt`, '--out', dir])
    return { dir, log, ...outputs(dir) }
  }
  throw new Error('not found')
}

const outputs = (dir: string) => {
  const files = fs.readdirSync(p(dir))
  return { gifs: files.filter((f) => f.endsWith('.gif')).sort().map((f) => `/${dir}/${f}`), grid: files.includes('grid.png') ? `/${dir}/grid.png` : null }
}

export function psgApi(): Plugin {
  return {
    name: 'psg-api',
    configureServer(server) {
      server.middlewares.use(async (req, res: ServerResponse, next) => {
        const url = (req.url ?? '').split('?')[0]
        if (url.startsWith('/out/')) { // generated images only; no path escapes
          const f = path.normalize(p(decodeURIComponent(url)))
          if (!f.startsWith(p('out') + path.sep) || !fs.existsSync(f)) return next()
          res.setHeader('Content-Type', f.endsWith('.gif') ? 'image/gif' : 'image/png')
          return fs.createReadStream(f).pipe(res)
        }
        if (!url.startsWith('/api/')) return next()
        res.setHeader('Content-Type', 'application/json')
        try { res.end(JSON.stringify(await api(url, req))) } catch (e) {
          res.statusCode = 400
          res.end(JSON.stringify({ error: (e as Error).message }))
        }
      })
    },
  }
}
