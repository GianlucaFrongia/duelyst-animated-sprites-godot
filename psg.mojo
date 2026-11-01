# psg: run training pipelines in the background and watch their progress.
#
#     mojo build -O3 psg.mojo -o psg
#     ./psg run pipelines/toy.pipe          # start a pipeline and watch it (Ctrl-C detaches)
#     ./psg train small -- --data data_duelyst --steps 4000 --dim 64
#     ./psg status                          # every run, once
#     ./psg watch [name]                    # live view
#     ./psg logs <name> [stage]             # tail of a stage's log
#     ./psg stop <name>
#     ./psg adopt <name> <pid> <log>        # track a training started by hand
#
# Runs are detached worker processes, so closing the terminal doesn't stop them. Stages
# that use the GPU wait for each other, so pipelines started together queue up.

from std.sys import argv
from std.subprocess import run
from std.os import listdir, makedirs, remove
from std.os.path import exists, isdir
from std.time import sleep
from src.pipeline import (
    Stage, Progress, parse_pipeline, substituted, parse_progress, stage_state, run_state,
    is_failure, has_ended, tool_bin, ckpt_arch,
    stage_file, read_text, read_or, write_text, q, alive, now_epoch,
)

comptime RUNS = "runs"
comptime BAR = 24


def esc(code: String) -> String:
    return chr(27) + "[" + code + "m"


def color(s: String, code: String) -> String:
    return esc(code) + s + esc("0")


def pad(s: String, n: Int) -> String:
    var out = s
    while out.byte_length() < n:
        out += " "
    return out^


def lpad(s: String, n: Int, fill: String = " ") -> String:
    var out = s
    while out.byte_length() < n:
        out = fill + out
    return out^


def fmt_float(x: Float64, digits: Int) -> String:
    """x with a fixed number of decimals."""
    var scale = 1
    for _ in range(digits):
        scale *= 10
    var v = Int(abs(x) * Float64(scale) + 0.5)
    var s = ("-" if x < 0 else "") + String(v // scale)
    if digits > 0:
        s += "." + lpad(String(v % scale), digits, "0")
    return s^


def fmt_rate(rate: String) -> String:
    """'12.422249 pairs/s' -> '12.4 pairs/s'."""
    var unit = String("")
    var num = String(rate.split(" ")[0])
    var parts = rate.split(" ")
    for i in range(1, len(parts)):
        unit += " " + String(parts[i])
    try:
        return fmt_float(Float64(num), 1) + unit
    except:
        return rate


def fmt_duration(secs: Int) -> String:
    if secs < 60:
        return String(secs) + "s"
    if secs < 3600:
        return String(secs // 60) + "m" + lpad(String(secs % 60), 2, "0") + "s"
    return String(secs // 3600) + "h" + lpad(String((secs % 3600) // 60), 2, "0") + "m"


def bar(frac: Float64) -> String:
    var f = max(0.0, min(1.0, frac))
    var full = Int(f * Float64(BAR))
    var s = String("")
    for i in range(BAR):
        s += "█" if i < full else "░"
    return s^


def sparkline(vals: List[Float64], width: Int) -> String:
    if len(vals) < 2:
        return ""
    var ticks = List[String]()
    for t in ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]:
        ticks.append(String(t))
    var start = max(0, len(vals) - width)
    var lo = vals[start]
    var hi = vals[start]
    for i in range(start, len(vals)):
        lo = min(lo, vals[i])
        hi = max(hi, vals[i])
    var s = String("")
    for i in range(start, len(vals)):
        var t = 0 if hi - lo < 1e-12 else Int((vals[i] - lo) / (hi - lo) * 7.0 + 0.5)
        s += ticks[t]
    return s^


def state_badge(s: String) -> String:
    if s == "running":
        return color("▶ running", "1;36")
    if s == "waiting":
        return color("⏸ waiting for GPU", "33")
    if s == "done":
        return color("✓ done", "32")
    if s == "skipped":
        return color("↷ skipped", "2")
    if s == "failed":
        return color("✗ failed", "1;31")
    if s == "died":
        return color("✗ died", "1;31")
    if s == "stopped":
        return color("■ stopped", "35")
    if s == "ended":
        return color("✓ ended", "32")
    if s == "queued":
        return color("… queued", "33")
    if s == "not run":
        return color("– not run", "2")
    return color("· pending", "2")


# ------------------------------------------------------------------ run directories


def basename(path: String) -> String:
    var parts = path.split("/")
    return String(parts[len(parts) - 1])


def run_names() raises -> List[String]:
    """Runs sorted by creation time (oldest first)."""
    var names = List[String]()
    var keys = List[String]()
    if not isdir(RUNS):
        return names^
    for e in listdir(RUNS):
        var n = String(e)
        if not isdir(RUNS + "/" + n) or not exists(RUNS + "/" + n + "/pipeline"):
            continue
        var key = meta_value(RUNS + "/" + n, "created") + n
        var at = len(names)
        while at > 0 and keys[at - 1] > key:
            at -= 1
        names.insert(at, n)
        keys.insert(at, key)
    return names^


def meta_value(rundir: String, key: String) raises -> String:
    for line in read_or(rundir + "/meta", "").splitlines():
        var l = String(line)
        if l.startswith(key + " "):
            return String(l[byte=key.byte_length() + 1:])
    return ""


def load_stages(rundir: String) raises -> List[Stage]:
    var name = String("")
    return parse_pipeline(read_text(rundir + "/pipeline"), name)


def new_run_dir(name: String) raises -> String:
    var n = name
    var k = 2
    while exists(RUNS + "/" + n):
        n = name + "-" + String(k)
        k += 1
    var d = RUNS + "/" + n
    makedirs(d, exist_ok=True)
    return d^


def require_run(name: String) raises -> String:
    var d = RUNS + "/" + name
    if not exists(d + "/pipeline"):
        raise Error("no run named '" + name + "' (see ./psg status)")
    return d^


# ------------------------------------------------------------------ rendering


def render_run(name: String, now: Int, detail: Bool) raises -> String:
    var d = RUNS + "/" + name
    var stages = load_stages(d)
    var state = run_state(d, len(stages))
    var created = meta_value(d, "created")
    var out = color(name, "1") + "  " + state_badge(state) + color("  started " + created, "2") + "\n"
    var finished = state == "done" or is_failure(state)
    for i in range(len(stages)):
        var s = stage_state(d, i)
        if s == "pending" and finished:
            s = "not run"
        var line = "  " + pad(stages[i].name, 12) + " "
        var below = String("")  # an extra line under the stage: loss sparkline or last log line
        if s == "running" or (has_ended(s) and exists(stage_file(d, i, "log"))):
            var p = parse_progress(read_text(stage_file(d, i, "log")))
            if p.total > 0:
                var frac = Float64(p.cur) / Float64(p.total)
                line += (color(bar(frac), "36") + " " + lpad(String(Int(frac * 100.0)), 3) + "%  "
                         + p.unit + " " + String(p.cur) + "/" + String(p.total))
                if p.loss >= 0.0:
                    line += "  loss " + fmt_float(p.loss, 4)
                if s == "running":
                    if p.rate.byte_length() > 0:
                        line += "  " + fmt_rate(p.rate)
                    if p.eta.byte_length() > 0:
                        line += "  eta " + p.eta
                line += "  "
                if detail and len(p.losses) > 1:
                    below = color("loss " + sparkline(p.losses, 48), "2")
            elif detail and s == "running" and p.last.byte_length() > 0:
                below = color(p.last if p.last.byte_length() <= 70 else String(p.last[byte=0:70]) + "…", "2")
        line += state_badge(s)
        var start = read_or(stage_file(d, i, "start"), "")
        if start.byte_length() > 0:
            var end = read_or(stage_file(d, i, "end"), "")
            var t1 = Int(end) if end.byte_length() > 0 else now
            line += color("  " + fmt_duration(t1 - Int(start)), "2")
        if below.byte_length() > 0:
            line += "\n  " + pad("", 12) + " " + below
        if s == "failed" or s == "died":
            line += color("  → ./psg logs " + name + " " + stages[i].name, "2")
        out += line + "\n"
    return out^


def print_status(only: String, detail: Bool) raises -> Bool:
    """Prints runs; returns True while any of them is still active."""
    var now = now_epoch()
    var names = run_names()
    if len(names) == 0:
        print("no runs yet: ./psg run pipelines/<file>.pipe")
        return False
    var active = False
    var text = String("")
    for n in names:
        if only.byte_length() > 0 and n != only:
            continue
        text += render_run(n, now, detail) + "\n"
        var d = RUNS + "/" + n
        var st = run_state(d, len(load_stages(d)))
        if st == "running" or st == "waiting" or st == "queued":
            active = True
    print(text, end="")
    return active


def watch(only: String, until_done: Bool) raises:
    while True:
        print(chr(27) + "[H" + chr(27) + "[2J", end="")  # home + clear
        print(color("psg watch", "1") + color("  (Ctrl-C to detach; runs keep going)", "2") + "\n")
        var active = print_status(only, True)
        if until_done and not active:
            return
        sleep(2.0)


# ------------------------------------------------------------------ commands


def project_root() raises -> String:
    return run("pwd -P")


def launch(rundir: String, pipeline_text: String, self_path: String, external: Bool) raises:
    var root = project_root()
    var name = String("")
    _ = parse_pipeline(pipeline_text, name)  # validate before anything starts
    write_text(rundir + "/pipeline", substituted(pipeline_text))
    var meta = "root " + root + "\ncreated " + run("date '+%Y-%m-%d %H:%M:%S'") + "\n"
    if external:
        meta += "external yes\n"
    write_text(rundir + "/meta", meta)
    if external:
        return
    var exe = self_path if self_path.startswith("/") else root + "/" + self_path
    # The worker ignores SIGINT/SIGHUP (inherited by the stages), so Ctrl-C in a watch or a
    # closed terminal doesn't kill training. `psg stop` uses SIGTERM.
    var pid = run("cd " + q(root) + "; (trap '' INT HUP; exec " + q(exe) + " _worker " + q(rundir)
                  + ") > " + q(rundir + "/worker.log") + " 2>&1 < /dev/null & echo $!")
    write_text(rundir + "/worker.pid", pid + "\n")


def gpu_busy(rundir: String) raises -> Bool:
    for n in run_names():
        var d = RUNS + "/" + n
        if d == rundir:
            continue
        var holder = read_or(d + "/gpu.lock", "")
        if holder.byte_length() > 0 and alive(holder):
            return True
    return False


def worker(rundir: String) raises:
    var root = meta_value(rundir, "root")
    var stages = load_stages(rundir)
    var me = run("echo $PPID")  # the shell's parent is this process
    for i in range(len(stages)):
        if exists(rundir + "/stopped"):
            return
        ref st = stages[i]
        if st.skip_if.byte_length() > 0 and exists(root + "/" + st.skip_if):
            write_text(stage_file(rundir, i, "skipped"), st.skip_if + " exists\n")
            write_text(stage_file(rundir, i, "rc"), "0\n")
            continue
        if st.gpu:
            write_text(stage_file(rundir, i, "waiting"), "")
            while True:
                while gpu_busy(rundir):
                    if exists(rundir + "/stopped"):
                        return
                    sleep(5.0)
                write_text(rundir + "/gpu.lock", me + "\n")
                sleep(0.5)
                if not gpu_busy(rundir):  # nobody grabbed it at the same moment
                    break
                remove(rundir + "/gpu.lock")
                sleep(3.0)
            remove(stage_file(rundir, i, "waiting"))
        write_text(stage_file(rundir, i, "start"), String(now_epoch()) + "\n")
        _ = run("cd " + q(root) + "; { " + st.cmd + " ; } > " + q(stage_file(rundir, i, "log"))
                + " 2>&1 < /dev/null & echo $! > " + q(stage_file(rundir, i, "pid"))
                + "; wait $!; echo $? > " + q(stage_file(rundir, i, "rc")))
        write_text(stage_file(rundir, i, "end"), String(now_epoch()) + "\n")
        if exists(rundir + "/gpu.lock"):
            remove(rundir + "/gpu.lock")
        if read_or(stage_file(rundir, i, "rc"), "1") != "0":
            return


def cmd_run(args: List[String], self_path: String) raises:
    if len(args) < 1:
        raise Error("usage: psg run <pipeline file> [--name NAME] [--detach]")
    var path = args[0]
    var text = read_text(path)
    var name = String("")
    _ = parse_pipeline(text, name)
    var detach = False
    for i in range(1, len(args)):
        if args[i] == "--name" and i + 1 < len(args):
            name = args[i + 1]
        if args[i] == "--detach":
            detach = True
    if name.byte_length() == 0:
        name = String(basename(path).split(".")[0])
    var d = new_run_dir(name)
    launch(d, text, self_path, False)
    print("started", d)
    if not detach:
        watch(basename(d), True)


def cmd_train(args: List[String], self_path: String) raises:
    """One training run as a one-stage pipeline: psg train NAME [--arch A] -- <train_gpu flags>.
    The architecture defaults to the --resume checkpoint's, else ctx."""
    if len(args) < 1:
        raise Error("usage: psg train <name> [--arch A] [--detach] -- --data <dir> [train_gpu flags]")
    var name = args[0]
    var arch = String("")
    var flags = String("")
    var detach = False
    var passthrough = False
    var i = 1
    while i < len(args):
        if passthrough:
            if args[i] == "--resume" and i + 1 < len(args) and arch.byte_length() == 0:
                arch = ckpt_arch(args[i + 1])
            flags += " " + q(args[i])
        elif args[i] == "--":
            passthrough = True
        elif args[i] == "--detach":
            detach = True
        elif args[i] == "--arch" and i + 1 < len(args):
            arch = args[i + 1]
            i += 1
        else:
            flags += " " + q(args[i])
        i += 1
    if arch.byte_length() == 0:
        arch = "ctx"
    var d = new_run_dir(name)
    var n = basename(d)
    var text = "stage train\n  cmd " + tool_bin("train_gpu", arch) + " --out checkpoints/" + n + flags + "\n"
    launch(d, text, self_path, False)
    print("started", d, "→ checkpoints/" + n)
    if not detach:
        watch(n, True)


def cmd_logs(args: List[String]) raises:
    if len(args) < 1:
        raise Error("usage: psg logs <name> [stage] [-n LINES]")
    var d = require_run(args[0])
    var stages = load_stages(d)
    var lines = 30
    var which = -1
    var i = 1
    while i < len(args):
        if args[i] == "-n" and i + 1 < len(args):
            lines = Int(args[i + 1])
            i += 1
        else:
            for j in range(len(stages)):
                if stages[j].name == args[i]:
                    which = j
        i += 1
    if which < 0:  # the latest stage that has a log
        for j in range(len(stages)):
            if exists(stage_file(d, j, "log")):
                which = j
    if which < 0:
        print("no logs yet")
        return
    var path = stage_file(d, which, "log")
    print(color("== " + stages[which].name + "  (" + path + ")", "2"))
    print(run("tail -n " + String(lines) + " " + q(path)))


def cmd_stop(args: List[String]) raises:
    if len(args) < 1:
        raise Error("usage: psg stop <name>")
    var d = require_run(args[0])
    write_text(d + "/stopped", "")
    var worker_pid = read_or(d + "/worker.pid", "")
    if alive(worker_pid):
        _ = run("kill -TERM " + worker_pid)
    var stages = load_stages(d)
    for i in range(len(stages)):
        var pid = read_or(stage_file(d, i, "pid"), "")
        if alive(pid):
            _ = run("pkill -TERM -P " + pid + "; kill -TERM " + pid)
            write_text(stage_file(d, i, "end"), String(now_epoch()) + "\n")
    if exists(d + "/gpu.lock"):
        remove(d + "/gpu.lock")
    print("stopped", args[0])


def cmd_adopt(args: List[String], self_path: String) raises:
    """Tracks a process started outside psg, e.g. a train_gpu you launched by hand."""
    if len(args) < 3:
        raise Error("usage: psg adopt <name> <pid> <log file>")
    if not alive(args[1]):
        raise Error("process " + args[1] + " is not running")
    var d = new_run_dir(args[0])
    launch(d, "stage train\n  cmd (started outside psg)\n  gpu yes\n", self_path, True)
    var log = args[2] if args[2].startswith("/") else project_root() + "/" + args[2]
    _ = run("ln -sf " + q(log) + " " + q(stage_file(d, 0, "log")))
    write_text(stage_file(d, 0, "pid"), args[1] + "\n")
    write_text(d + "/gpu.lock", args[1] + "\n")  # other gpu stages queue behind it
    write_text(stage_file(d, 0, "start"), run("ps -o lstart= -p " + args[1] + " | xargs -I{} date -j -f '%a %b %d %T %Y' '{}' +%s") + "\n")
    print("adopted", args[1], "as", d)


def usage():
    print("""psg: pipelines for pixel-sprite-gen

  ./psg run <file.pipe> [--name N] [--detach]   start a pipeline, then watch it
  ./psg train <name> [--arch A] [--detach] -- <train_gpu flags>
                                                one training run (checkpoints/<name>)
  ./psg status                                  all runs, once
  ./psg watch [name]                            live view (Ctrl-C detaches)
  ./psg logs <name> [stage] [-n 30]             tail a stage's log
  ./psg stop <name>                             stop a run and its current stage
  ./psg adopt <name> <pid> <log>                track a training started by hand

Pipeline files: see pipelines/*.pipe""")


def main() raises:
    var a = argv()
    var args = List[String]()
    for i in range(2, len(a)):
        args.append(String(a[i]))
    var self_path = String(a[0])
    if len(a) < 2:
        usage()
        return
    var cmd = String(a[1])
    try:
        if cmd == "run":
            cmd_run(args, self_path)
        elif cmd == "train":
            cmd_train(args, self_path)
        elif cmd == "status" or cmd == "ls":
            _ = print_status("", False)
        elif cmd == "watch":
            watch(args[0] if len(args) > 0 else "", False)
        elif cmd == "logs":
            cmd_logs(args)
        elif cmd == "stop":
            cmd_stop(args)
        elif cmd == "adopt":
            cmd_adopt(args, self_path)
        elif cmd == "_worker":
            worker(args[0])
        else:
            usage()
    except e:
        print(color("error: ", "1;31") + String(e))
