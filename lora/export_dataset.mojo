# Export the Duelyst sprites in this repo as a captioned LoRA training dataset.
#
# Reads the Godot SpriteFrames (.tres) files, cuts every animation out of its
# spritesheet and writes:
#   <out>/wan/videos/<id>.mp4 + <id>.txt   clips for Wan LoRA training
#   <out>/flux/grids/<id>.png + <id>.txt   frame grids for FLUX.2 LoRA training
#   <out>/frames/<id>/000.png ...          raw frames (only with --frames)
#   <out>/metadata.jsonl                   one JSON record per animation
#
# Image I/O goes through Python (Pillow, NumPy, imageio) via Mojo's Python interop.
#
# Usage:
#   mojo run lora/export_dataset.mojo [--out DIR] [--trigger WORD]
#       [--video-size 512] [--cell-size 256] [--grid-max 1024] [--fps 9] [--bg 64,64,64]
#       [--no-fx] [--no-video] [--no-grid] [--frames] [--limit N] [--only SUBSTR]

from std.python import Python, PythonObject
from std.sys import argv
from std.collections import Dict, List


comptime ASSETS = "addons/duelyst_animated_sprites/assets"


@fieldwise_init
struct Rect(Copyable, Movable):
    var x: Int
    var y: Int
    var w: Int
    var h: Int


@fieldwise_init
struct Animation(Copyable, Movable):
    var name: String
    var frames: List[Rect]
    var fps: Float64


@fieldwise_init
struct Config(Copyable, Movable):
    var root: String
    var out: String
    var trigger: String
    var video_size: Int
    var cell_size: Int
    var grid_max: Int
    var fps: Int
    var bg: String
    var include_fx: Bool
    var write_video: Bool
    var write_grid: Bool
    var write_frames: Bool
    var limit: Int
    var only: String


# --------------------------------------------------------------------------- #
# .tres parsing
# --------------------------------------------------------------------------- #


def between(s: String, start: String, end: String) raises -> String:
    """Text between the first `start` and the next `end` after it ("" if absent)."""
    var a = s.find(start)
    if a < 0:
        return ""
    a += start.byte_length()
    var b = s.find(end, a)
    if b < 0:
        return ""
    return String(s[byte=a:b])


def parse_rect(line: String) raises -> Rect:
    var nums = between(line, "Rect2(", ")").split(",")
    return Rect(
        Int(nums[0].strip()),
        Int(nums[1].strip()),
        Int(nums[2].strip()),
        Int(nums[3].strip()),
    )


def parse_tres(
    text: String, mut sheet_path: String, mut anims: List[Animation]
) raises:
    """Fill `sheet_path` (res:// path of the atlas) and `anims` from a .tres file.

    Frame order is taken from the `animations` array, where each animation lists
    its frames and then its name.
    """
    var regions = Dict[String, Rect]()
    var current_id = String("")
    var pending = List[Rect]()
    var pending_fps = 9.0
    for raw in text.split("\n"):
        var line = String(raw.strip())
        if line.startswith("[ext_resource") and sheet_path == "":
            sheet_path = between(line, 'path="', '"')
        elif line.startswith("[sub_resource"):
            current_id = between(line, 'id="', '"')
        elif line.startswith("region = Rect2("):
            regions[current_id] = parse_rect(line)
        elif line.startswith('"texture": SubResource("'):
            var id = between(line, 'SubResource("', '"')
            if id in regions:
                pending.append(regions[id].copy())
        elif line.startswith('"name": &"'):
            anims.append(
                Animation(between(line, '&"', '"'), pending.copy(), pending_fps)
            )
            pending = List[Rect]()
        elif line.startswith('"speed":'):
            # speed comes after name; patch the animation just added.
            var v = String(line[byte = 8 :]).strip()
            if len(anims) > 0:
                anims[len(anims) - 1].fps = Float64(String(v))


# --------------------------------------------------------------------------- #
# Captions
# --------------------------------------------------------------------------- #


def faction_of(prefix: String) -> String:
    if prefix == "f1":
        return "Lyonar faction"
    if prefix == "f2":
        return "Songhai faction"
    if prefix == "f3":
        return "Vetruvian faction"
    if prefix == "f4":
        return "Abyssian faction"
    if prefix == "f5":
        return "Magmar faction"
    if prefix == "f6":
        return "Vanar faction"
    if prefix == "boss":
        return "boss monster"
    if prefix == "neutral":
        return "neutral faction"
    return ""


def normalize_action(name: String) -> String:
    var n = name.lower()
    if n.startswith("breath"):
        return "breathing"
    if n == "die" or n.startswith("death"):
        return "death"
    if n == "hurt" or n == "damage":
        return "hit"
    if n == "move" or n == "movement" or n == "crawl":
        return "run"
    if n.startswith("cast"):
        return "cast"
    return n


def describe_action(action: String) -> String:
    if action == "attack":
        return "attack animation"
    if action == "idle":
        return "idle animation"
    if action == "breathing":
        return "breathing idle loop"
    if action == "run":
        return "running animation"
    if action == "hit":
        return "taking a hit, hurt reaction animation"
    if action == "death":
        return "death animation"
    if action == "cast":
        return "spell casting animation"
    if action == "projectile":
        return "projectile animation"
    return action + " animation"


def any_in(name: String, words: List[String]) -> Bool:
    for w in words:
        if w in name:
            return True
    return False


def element_tags(id: String) -> List[String]:
    """Element tags guessed from keywords in the asset name.

    This is a heuristic: it only sees the name, not the pixels. Refine the
    captions later (e.g. with a vision model) for better prompt following.
    """
    var n = id.lower()
    var tags = List[String]()
    if any_in(n, ["fire", "flame", "lava", "magma", "burn", "ember", "inferno",
                  "phoenix", "blaze", "pyre", "scorch", "molten", "volcan",
                  "cinder", "firebreath"]):
        tags.append("fire")
    if any_in(n, ["ice", "frost", "snow", "winter", "frozen", "glacier", "cryo",
                  "hail", "chill", "freeze", "boreal", "blizzard"]):
        tags.append("ice")
    if any_in(n, ["storm", "thunder", "lightning", "tesla", "spark", "volt",
                  "shock", "static", "plasma"]):
        tags.append("lightning")
    var no_lightning = n.replace("lightning", "")
    if any_in(no_lightning, ["holy", "radiant", "divine", "sun", "solar", "light",
                             "dawn", "aurora"]):
        tags.append("holy light")
    if any_in(n, ["shadow", "dark", "night", "void", "shade", "abyss", "gloom",
                  "wraith", "nether", "doom"]):
        tags.append("shadow")
    if any_in(n, ["poison", "venom", "toxic", "acid", "plague", "blight"]):
        tags.append("poison")
    if any_in(n, ["water", "tide", "wave", "ocean", "aqua", "splash"]):
        tags.append("water")
    if any_in(n, ["stone", "rock", "earth", "sand", "dune", "boulder", "crystal",
                  "quake"]):
        tags.append("earth")
    if any_in(n, ["wind", "gale", "tempest", "tornado", "cyclone"]):
        tags.append("wind")
    if "blood" in n:
        tags.append("blood")
    return tags^


def element_phrase(tag: String) -> String:
    if tag == "fire":
        return "fire element, flames"
    if tag == "ice":
        return "ice element, frost"
    if tag == "lightning":
        return "lightning element, electric sparks"
    if tag == "holy light":
        return "holy light element, glowing"
    if tag == "shadow":
        return "shadow element, dark energy"
    return tag + " element"


def human_name(id: String) -> String:
    """`f5_firebreather` -> `firebreather`, `neutral_mercsunseer` -> `mercsunseer`."""
    var parts = id.split("_")
    var words = List[String]()
    for i in range(1, len(parts)):
        words.append(String(parts[i]))
    if len(words) == 0:
        return id
    return String(" ").join(words)


def make_caption(
    trigger: String,
    kind: String,
    id: String,
    anim_name: String,
    n_frames: Int,
    cols: Int,
    rows: Int,
    bg_word: String,
) raises -> String:
    var tags = element_tags(id)
    var bits = List[String]()
    bits.append(trigger)
    if kind == "grid":
        bits.append(
            "pixel art spritesheet, "
            + String(cols)
            + "x"
            + String(rows)
            + " grid of "
            + String(n_frames)
            + " animation frames, read left to right, top to bottom"
        )
    else:
        bits.append("pixel art sprite animation")

    var prefix = String(id.split("_")[0])
    if prefix == "fx" or "_fx_" in id:
        bits.append("visual effect animation")
        bits.append(human_name(id).replace("fx ", ""))
    else:
        bits.append(describe_action(normalize_action(anim_name)))
        bits.append(human_name(id))
        var faction = faction_of(prefix)
        if faction != "":
            bits.append(faction)
    for t in tags:
        bits.append(element_phrase(t))
    if not (prefix == "fx" or "_fx_" in id):
        bits.append("side view, facing right")
    bits.append("plain " + bg_word + " background")
    return String(", ").join(bits)


# --------------------------------------------------------------------------- #
# Image work (Python interop)
# --------------------------------------------------------------------------- #


def canvas_for(max_side: Int) -> Int:
    """Square canvas that fits the frame, rounded up to a multiple of 8."""
    return max(16, ((max_side + 7) // 8) * 8)


def tight_frames(sheet: PythonObject, frames: List[Rect], margin: Int) raises -> List[Rect]:
    """Crop every frame of an animation to the union of their opaque bounding
    boxes. Using one shared box keeps the character aligned between frames
    while dropping the empty space around it."""
    var x0 = 1 << 30
    var y0 = 1 << 30
    var x1 = -1
    var y1 = -1
    for r in frames:
        var bbox = sheet.crop(Python.tuple(r.x, r.y, r.x + r.w, r.y + r.h)).getbbox()
        if bbox is None:
            continue
        x0 = min(x0, Int(py=bbox[0]))
        y0 = min(y0, Int(py=bbox[1]))
        x1 = max(x1, Int(py=bbox[2]))
        y1 = max(y1, Int(py=bbox[3]))
    var out = List[Rect]()
    if x1 < 0:
        return out^
    for r in frames:
        var lx = max(0, x0 - margin)
        var ly = max(0, y0 - margin)
        var rx = min(r.w, x1 + margin)
        var ry = min(r.h, y1 + margin)
        out.append(Rect(r.x + lx, r.y + ly, rx - lx, ry - ly))
    return out^


def place_frames(
    pil: PythonObject,
    sheet: PythonObject,
    frames: List[Rect],
    canvas: Int,
    scale: Int,
    out_side: Int,
    bg: PythonObject,
) raises -> PythonObject:
    """Crop each frame, centre it on a canvas, nearest-neighbour upscale by an
    integer factor (keeps the pixel grid crisp) and flatten onto `bg`."""
    var result = Python.list()
    for r in frames:
        var crop = sheet.crop(Python.tuple(r.x, r.y, r.x + r.w, r.y + r.h))
        var tile = pil.new("RGBA", Python.tuple(canvas, canvas), Python.tuple(0, 0, 0, 0))
        tile.paste(crop, Python.tuple((canvas - r.w) // 2, (canvas - r.h) // 2))
        var big = tile.resize(Python.tuple(canvas * scale, canvas * scale), pil.NEAREST)
        var flat = pil.new("RGBA", Python.tuple(out_side, out_side), bg)
        var off = (out_side - canvas * scale) // 2
        flat.alpha_composite(big, Python.tuple(off, off))
        result.append(flat.convert("RGB"))
    return result


def main() raises:
    var cfg = Config(
        "data/duelyst", "lora/dataset", "dlystsprite", 512, 256, 1024, 9, "64,64,64",
        True, True, True, False, 0, "",
    )
    var args = argv()
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--out":
            i += 1; cfg.out = String(args[i])
        elif a == "--root":
            i += 1; cfg.root = String(args[i])
        elif a == "--trigger":
            i += 1; cfg.trigger = String(args[i])
        elif a == "--video-size":
            i += 1; cfg.video_size = Int(String(args[i]))
        elif a == "--cell-size":
            i += 1; cfg.cell_size = Int(String(args[i]))
        elif a == "--grid-max":
            i += 1; cfg.grid_max = Int(String(args[i]))
        elif a == "--fps":
            i += 1; cfg.fps = Int(String(args[i]))
        elif a == "--bg":
            i += 1; cfg.bg = String(args[i])
        elif a == "--limit":
            i += 1; cfg.limit = Int(String(args[i]))
        elif a == "--only":
            i += 1; cfg.only = String(args[i])
        elif a == "--no-fx":
            cfg.include_fx = False
        elif a == "--no-video":
            cfg.write_video = False
        elif a == "--no-grid":
            cfg.write_grid = False
        elif a == "--frames":
            cfg.write_frames = True
        else:
            raise Error("unknown argument: " + a)
        i += 1

    var os = Python.import_module("os")
    var json = Python.import_module("json")
    var math = Python.import_module("math")
    var pil = Python.import_module("PIL.Image")
    var imageio: PythonObject = Python.none()
    if cfg.write_video:
        imageio = Python.import_module("imageio.v2")

    var bgp = cfg.bg.split(",")
    var bg = Python.tuple(Int(bgp[0].strip()), Int(bgp[1].strip()), Int(bgp[2].strip()), 255)
    var bg_word = String("dark gray") if cfg.bg == "64,64,64" else String("solid color")

    var video_dir = cfg.out + "/wan/videos"
    var grid_dir = cfg.out + "/flux/grids"
    if cfg.write_video:
        os.makedirs(video_dir, exist_ok=True)
    if cfg.write_grid:
        os.makedirs(grid_dir, exist_ok=True)
    os.makedirs(cfg.out, exist_ok=True)
    var meta = Python.evaluate("open")(cfg.out + "/metadata.jsonl", "w")

    var groups = List[String]()
    groups.append("units")
    if cfg.include_fx:
        groups.append("fx")

    var n_written = 0
    var n_skipped = 0
    for group in groups:
        var frames_dir = cfg.root + "/" + ASSETS + "/spriteframes/" + group
        var files = Python.evaluate("sorted")(os.listdir(frames_dir))
        for f in files:
            var fname = String(f)
            if not fname.endswith(".tres"):
                continue
            var id = String(fname[byte = 0 : fname.byte_length() - 5])
            if cfg.only != "" and not (cfg.only in id):
                continue
            if cfg.limit > 0 and n_written >= cfg.limit:
                break

            var text = String(Python.evaluate("lambda p: open(p).read()")(frames_dir + "/" + fname))
            var sheet_res = String("")
            var anims = List[Animation]()
            parse_tres(text, sheet_res, anims)
            if sheet_res == "" or len(anims) == 0:
                continue
            var sheet_path = cfg.root + "/" + sheet_res.replace("res://", "")
            var sheet = pil.open(sheet_path).convert("RGBA")

            for anim in anims:
                var n = len(anim.frames)
                var frames = tight_frames(sheet, anim.frames, 2)
                var max_side = 0
                for r in frames:
                    max_side = max(max_side, max(r.w, r.h))
                # Skip empty animations and tiny projectile sprites.
                if n < 2 or len(frames) == 0 or max_side < 24:
                    n_skipped += 1
                    continue

                var clip_id = id + "__" + anim.name
                var canvas = canvas_for(max_side)
                var record = Python.dict()
                record["id"] = clip_id
                record["asset"] = id
                record["group"] = group
                record["animation"] = anim.name
                record["action"] = normalize_action(anim.name)
                record["frames"] = n
                record["frame_size"] = max_side
                record["source_fps"] = anim.fps
                var tag_list = Python.list()
                for t in element_tags(id):
                    tag_list.append(t)
                record["elements"] = tag_list

                if cfg.write_video:
                    var vscale = max(1, cfg.video_size // canvas)
                    var vframes = place_frames(pil, sheet, frames, canvas, vscale, cfg.video_size, bg)
                    var np = Python.import_module("numpy")
                    var arrays = Python.list()
                    for fr in vframes:
                        arrays.append(np.asarray(fr))
                    # Near-lossless H.264 so the hard pixel edges survive.
                    imageio.mimsave(
                        video_dir + "/" + clip_id + ".mp4", arrays,
                        fps=cfg.fps, codec="libx264", macro_block_size=16,
                        output_params=Python.list("-crf", "4"),
                    )
                    var vcap = make_caption(cfg.trigger, "video", id, anim.name, n, 0, 0, bg_word)
                    Python.evaluate("lambda p, s: open(p, 'w').write(s)")(video_dir + "/" + clip_id + ".txt", vcap)
                    record["video"] = "wan/videos/" + clip_id + ".mp4"
                    record["video_scale"] = vscale
                    record["video_caption"] = vcap

                if cfg.write_grid:
                    var cols = Int(py=math.ceil(math.sqrt(n)))
                    var rows = (n + cols - 1) // cols
                    # Keep the whole sheet within --grid-max so the trainer never has
                    # to resize it (which would blur the pixel grid).
                    var cell = min(cfg.cell_size, (cfg.grid_max // cols) // 16 * 16)
                    cell = max(cell, canvas)
                    var gscale = max(1, cell // canvas)
                    var gframes = place_frames(pil, sheet, frames, canvas, gscale, cell, bg)
                    var grid = pil.new("RGB", Python.tuple(cols * cell, rows * cell), Python.tuple(bg[0], bg[1], bg[2]))
                    for k in range(n):
                        grid.paste(gframes[k], Python.tuple((k % cols) * cell, (k // cols) * cell))
                    grid.save(grid_dir + "/" + clip_id + ".png")
                    var gcap = make_caption(cfg.trigger, "grid", id, anim.name, n, cols, rows, bg_word)
                    Python.evaluate("lambda p, s: open(p, 'w').write(s)")(grid_dir + "/" + clip_id + ".txt", gcap)
                    record["grid"] = "flux/grids/" + clip_id + ".png"
                    record["grid_cols"] = cols
                    record["grid_rows"] = rows
                    record["grid_scale"] = gscale
                    record["grid_caption"] = gcap

                if cfg.write_frames:
                    var fdir = cfg.out + "/frames/" + clip_id
                    os.makedirs(fdir, exist_ok=True)
                    var k = 0
                    for r in anim.frames:
                        sheet.crop(Python.tuple(r.x, r.y, r.x + r.w, r.y + r.h)).save(
                            fdir + "/" + String(PythonObject(k).__str__().zfill(3)) + ".png"
                        )
                        k += 1

                meta.write(json.dumps(record) + "\n")
                n_written += 1
                if n_written % 100 == 0:
                    print("  exported", n_written, "animations...")
                if cfg.limit > 0 and n_written >= cfg.limit:
                    break

    meta.close()
    print("Done:", n_written, "animations exported to", cfg.out, "(skipped", n_skipped, "tiny/empty)")
