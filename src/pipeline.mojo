# Pipeline files, run directories and log parsing for the psg CLI.
#
# A pipeline is a text file of stages, each a shell command run from the project root:
#
#     set DATA data_duelyst              # ${DATA} is substituted in later lines
#     stage prepare
#       cmd .venv/bin/python tools/prepare_animations.py --atlases raw/units --out ${DATA}
#       skip_if ${DATA}/meta.txt          # skip the stage when this path exists
#     stage train
#       cmd ./train_gpu --data ${DATA} --out checkpoints/small --steps 4000
#       gpu yes                           # wait until no other run is using the GPU
#
# Each run lives in runs/<name>/. Everything psg shows is derived from the files there,
# so `status` works without talking to the worker process:
#     pipeline      the pipeline text, variables already substituted
#     meta          "root <dir>", "created <time>", optional "external yes"
#     worker.pid    pid of the detached worker that runs the stages
#     gpu.lock      pid holding the GPU while a gpu stage runs
#     stopped       written by `psg stop`
#     <i>.log .pid .rc .start .end .skipped .waiting    per stage i

from std.subprocess import run
from std.os.path import exists


struct Stage(Copyable, Movable):
    var name: String
    var cmd: String
    var skip_if: String
    var gpu: Bool
    var gpu_set: Bool

    def __init__(out self, name: String):
        self.name = name
        self.cmd = ""
        self.skip_if = ""
        self.gpu = False
        self.gpu_set = False


def read_text(path: String) raises -> String:
    var f = open(path, "r")
    var t = f.read()
    f.close()
    return t^


def read_or(path: String, default: String) raises -> String:
    if not exists(path):
        return default
    return String(read_text(path).strip())


def write_text(path: String, text: String) raises:
    var f = open(path, "w")
    f.write(text)
    f.close()


def q(s: String) -> String:
    """Single-quotes s for the shell."""
    return "'" + s.replace("'", "'\\''") + "'"


def head_rest(line: String, mut rest: String) -> String:
    """Splits 'key the rest' into the key (returned) and the rest."""
    var parts = line.split(" ")
    rest = ""
    for i in range(1, len(parts)):
        if i > 1:
            rest += " "
        rest += String(parts[i])
    return String(parts[0])


def parse_pipeline(text: String, mut name: String) raises -> List[Stage]:
    var stages = List[Stage]()
    var keys = List[String]()
    var vals = List[String]()
    var lineno = 0
    for raw in text.splitlines():
        lineno += 1
        var line = String(raw.strip())
        if line.byte_length() == 0 or line.startswith("#"):
            continue
        for i in range(len(keys)):
            line = line.replace("${" + keys[i] + "}", vals[i])
        var raw_rest = String("")
        var key = head_rest(line, raw_rest)
        var rest = String(raw_rest.strip())
        if key == "set":
            var value = String("")
            var k = head_rest(rest, value)
            keys.append(k)
            vals.append(String(value.strip()))
        elif key == "name":
            name = rest
        elif key == "stage":
            stages.append(Stage(rest if rest.byte_length() > 0 else "stage" + String(len(stages))))
        elif len(stages) == 0:
            raise Error("line " + String(lineno) + ": '" + key + "' before the first 'stage'")
        elif key == "cmd":
            ref s = stages[len(stages) - 1]
            s.cmd = rest if s.cmd.byte_length() == 0 else s.cmd + " && " + rest
        elif key == "skip_if":
            stages[len(stages) - 1].skip_if = rest
        elif key == "gpu":
            stages[len(stages) - 1].gpu = rest == "yes" or rest == "true"
            stages[len(stages) - 1].gpu_set = True
        else:
            raise Error("line " + String(lineno) + ": unknown key '" + key + "'")
    for i in range(len(stages)):
        ref s = stages[i]
        if s.cmd.byte_length() == 0:
            raise Error("stage '" + s.name + "' has no cmd")
        if not s.gpu_set:  # GPU programs in this repo, unless the stage says otherwise
            s.gpu = "train_gpu" in s.cmd or "./sample" in s.cmd or "gpu_check" in s.cmd
    return stages^


def substituted(text: String) raises -> String:
    """The pipeline text with ${VAR}s expanded, so a run keeps exactly what it ran."""
    var keys = List[String]()
    var vals = List[String]()
    var out = String("")
    for raw in text.splitlines():
        var line = String(raw)
        for i in range(len(keys)):
            line = line.replace("${" + keys[i] + "}", vals[i])
        var rest = String("")
        var key = head_rest(String(line.strip()), rest)
        if key == "set":
            var value = String("")
            var k = head_rest(String(rest.strip()), value)
            keys.append(k)
            vals.append(String(value.strip()))
        out += line + "\n"
    return out^


# ------------------------------------------------------------------ processes


def alive(pid: String) raises -> Bool:
    if pid.byte_length() == 0:
        return False
    for c in pid.codepoints():
        if c < Codepoint.ord("0") or c > Codepoint.ord("9"):
            return False
    return run("kill -0 " + pid + " 2>/dev/null && echo 1") == "1"


def now_epoch() raises -> Int:
    return Int(run("date +%s"))


# ------------------------------------------------------------------ run state


def stage_file(rundir: String, i: Int, ext: String) -> String:
    return rundir + "/" + String(i) + "." + ext


def stage_state(rundir: String, i: Int) raises -> String:
    """pending | waiting | running | done | skipped | failed | stopped | died | ended"""
    if exists(stage_file(rundir, i, "skipped")):
        return "skipped"
    var rc = read_or(stage_file(rundir, i, "rc"), "")
    if rc.byte_length() > 0:
        if rc == "0":
            return "done"
        return "stopped" if exists(rundir + "/stopped") else "failed"
    var pid = read_or(stage_file(rundir, i, "pid"), "")
    if pid.byte_length() > 0:
        if alive(pid):
            return "running"
        if exists(rundir + "/stopped"):
            return "stopped"
        return "ended" if read_or(rundir + "/meta", "").find("external yes") >= 0 else "died"
    if exists(rundir + "/stopped"):
        return "stopped"
    var worker = read_or(rundir + "/worker.pid", "")
    if exists(stage_file(rundir, i, "waiting")) and alive(worker):
        return "waiting"
    return "pending"


def run_state(rundir: String, n: Int) raises -> String:
    for i in range(n):
        var s = stage_state(rundir, i)
        if s == "failed" or s == "stopped" or s == "died":
            return s
        if s == "running" or s == "waiting":
            return s
        if s == "pending":
            # Pending with a live worker means it is between stages.
            return "running" if alive(read_or(rundir + "/worker.pid", "")) else "queued"
        if s == "ended":
            return "ended"
    return "done"


# ------------------------------------------------------------------ progress


struct Progress(Movable):
    var cur: Int
    var total: Int
    var unit: String  # "step", "sheets", ...
    var loss: Float64
    var rate: String
    var eta: String
    var losses: List[Float64]
    var last: String  # last non-empty log line

    def __init__(out self):
        self.cur = -1
        self.total = 0
        self.unit = ""
        self.loss = -1.0
        self.rate = ""
        self.eta = ""
        self.losses = List[Float64]()
        self.last = ""


def parse_progress(log: String) -> Progress:
    """Reads lines like 'step 900 / 20000 | loss 0.85 | ... | 12.4 pairs/s | eta 204 min'
    (train_gpu) or 'sheets 1 / 4 in 3.2 s' (sample)."""
    var p = Progress()
    for raw in log.splitlines():
        var line = String(raw.strip())
        if line.byte_length() == 0:
            continue
        p.last = line
        var parts = line.split(" | ")
        var tok = String(parts[0]).split(" ")
        if len(tok) < 4 or String(tok[2]) != "/":
            continue
        try:
            p.cur = Int(String(tok[1]))
            p.total = Int(String(tok[3]))
            p.unit = String(tok[0])
        except:
            continue
        for j in range(1, len(parts)):
            var f = String(parts[j])
            try:
                if f.startswith("loss "):
                    var rest = String("")
                    _ = head_rest(f, rest)
                    p.loss = Float64(rest)
                    p.losses.append(p.loss)
                elif f.startswith("eta "):
                    var rest = String("")
                    _ = head_rest(f, rest)
                    p.eta = rest
                elif f.endswith("/s"):
                    p.rate = f
            except:
                pass
    return p^
