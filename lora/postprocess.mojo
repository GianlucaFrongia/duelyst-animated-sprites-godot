# Turn generated output (a Wan video or a FLUX.2 spritesheet grid) back into
# crisp, Godot-ready pixel art.
#
# Steps per animation:
#   1. split into frames (video frames, or the cells of a --grid CxR sheet)
#   2. downscale by --factor (the integer upscale used for training data)
#   3. key out the flat background with a flood fill from the frame borders,
#      so dark holes inside the sprite stay opaque (as in the source art)
#   4. quantize all frames to one shared --colors palette (no dithering)
#   5. drop empty frames, crop all frames to one shared bounding box
#   6. write frames/, a horizontal strip PNG and a Godot SpriteFrames .tres
#
# Usage:
#   mojo run lora/postprocess.mojo INPUT --out DIR [--name attack]
#       [--grid 5x5] [--factor 4] [--colors 64] [--bg 64,64,64] [--tol 16]
#       [--fps 9] [--res-dir res://generated/]
#
# INPUT is a video (.mp4/.webm/.gif), an image (with --grid for sheets) or a
# directory of frame images.

from std.python import Python, PythonObject
from std.sys import argv
from std.collections import List


def flood_background(
    rgb: PythonObject, w: Int, h: Int, bg_r: Int, bg_g: Int, bg_b: Int, tol: Int
) raises -> PythonObject:
    """Return a Python bytearray alpha mask (0 = background, 255 = sprite).

    Background is every pixel within `tol` (max channel difference) of the
    background colour that is connected to the frame border. The breadth-first
    search runs in Mojo over a flat buffer.
    """
    var np = Python.import_module("numpy")
    var diff = np.abs(rgb.astype("int16") - np.array(Python.list(bg_r, bg_g, bg_b), dtype="int16")).max(axis=2)
    var near_py = (diff <= tol).astype("uint8").ravel().tobytes()
    var n = w * h
    var near = List[UInt8](capacity=n)
    for b in near_py:
        near.append(UInt8(Int(py=b)))

    var is_bg = List[UInt8](length=n, fill=0)
    var queue = List[Int](capacity=n)
    for x in range(w):
        queue.append(x)
        queue.append((h - 1) * w + x)
    for y in range(h):
        queue.append(y * w)
        queue.append(y * w + w - 1)

    var head = 0
    while head < len(queue):
        var p = queue[head]
        head += 1
        if is_bg[p] == 1 or near[p] == 0:
            continue
        is_bg[p] = 1
        var x = p % w
        var y = p // w
        if x > 0:
            queue.append(p - 1)
        if x < w - 1:
            queue.append(p + 1)
        if y > 0:
            queue.append(p - w)
        if y < h - 1:
            queue.append(p + w)

    var alpha = Python.evaluate("bytearray")(n)
    for i in range(n):
        if is_bg[i] == 0:
            alpha[i] = 255
    return alpha


def main() raises:
    var input = String("")
    var out = String("lora/generated")
    var name = String("attack")
    var grid_cols = 0
    var grid_rows = 0
    var factor = 4
    var colors = 64
    var bg = String("64,64,64")
    var tol = 16
    var fps = 9
    var res_dir = String("res://generated/")

    var args = argv()
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--out":
            i += 1; out = String(args[i])
        elif a == "--name":
            i += 1; name = String(args[i])
        elif a == "--grid":
            i += 1
            var g = String(args[i]).lower().split("x")
            grid_cols = Int(g[0]); grid_rows = Int(g[1])
        elif a == "--factor":
            i += 1; factor = Int(String(args[i]))
        elif a == "--colors":
            i += 1; colors = Int(String(args[i]))
        elif a == "--bg":
            i += 1; bg = String(args[i])
        elif a == "--tol":
            i += 1; tol = Int(String(args[i]))
        elif a == "--fps":
            i += 1; fps = Int(String(args[i]))
        elif a == "--res-dir":
            i += 1; res_dir = String(args[i])
        elif a.startswith("--"):
            raise Error("unknown argument: " + a)
        else:
            input = a
        i += 1
    if input == "":
        raise Error("usage: postprocess.mojo INPUT --out DIR [--grid CxR] [--factor 4] ...")
    if not res_dir.endswith("/"):
        res_dir += "/"

    var os = Python.import_module("os")
    var np = Python.import_module("numpy")
    var pil = Python.import_module("PIL.Image")
    var bgp = bg.split(",")
    var bg_r = Int(bgp[0].strip())
    var bg_g = Int(bgp[1].strip())
    var bg_b = Int(bgp[2].strip())

    # 1. Load frames as RGB PIL images.
    var frames = Python.list()
    var lower = input.lower()
    if os.path.isdir(input):
        for f in Python.evaluate("sorted")(os.listdir(input)):
            var fl = String(f).lower()
            if fl.endswith(".png") or fl.endswith(".jpg") or fl.endswith(".webp"):
                frames.append(pil.open(input + "/" + String(f)).convert("RGB"))
    elif lower.endswith(".mp4") or lower.endswith(".webm") or lower.endswith(".gif") or lower.endswith(".mov"):
        var iio = Python.import_module("imageio.v2")
        for arr in iio.mimread(input, memtest=False):
            frames.append(pil.fromarray(np.asarray(arr)[:, :, :3]))
    else:
        var img = pil.open(input).convert("RGB")
        if grid_cols > 0:
            var cw = Int(py=img.width) // grid_cols
            var ch = Int(py=img.height) // grid_rows
            for r in range(grid_rows):
                for c in range(grid_cols):
                    frames.append(img.crop(Python.tuple(c * cw, r * ch, (c + 1) * cw, (r + 1) * ch)))
        else:
            frames.append(img)
    if len(frames) == 0:
        raise Error("no frames found in " + input)

    # 2. Downscale. BOX averages each factor x factor block, which recovers the
    #    intended pixel from slightly blurry generated output.
    var small = Python.list()
    for f in frames:
        var w = max(1, Int(py=f.width) // factor)
        var h = max(1, Int(py=f.height) // factor)
        small.append(f.resize(Python.tuple(w, h), pil.BOX))

    # 3. Key out the background *before* quantizing, so dark sprite colours
    #    can't be merged into the background colour and then erased.
    var alphas = Python.list()
    for f in small:
        var w = Int(py=f.width)
        var h = Int(py=f.height)
        alphas.append(pil.frombytes("L", Python.tuple(w, h), Python.evaluate("bytes")(
            flood_background(np.asarray(f), w, h, bg_r, bg_g, bg_b, tol))))

    # 4. One palette for all frames so colours don't flicker between frames.
    var total_w = 0
    var max_h = 0
    for f in small:
        total_w += Int(py=f.width)
        max_h = max(max_h, Int(py=f.height))
    var atlas = pil.new("RGB", Python.tuple(total_w, max_h), Python.tuple(bg_r, bg_g, bg_b))
    var x = 0
    for f in small:
        atlas.paste(f, Python.tuple(x, 0))
        x += Int(py=f.width)
    var pal = atlas.quantize(colors=colors, method=pil.Quantize.MEDIANCUT, dither=pil.Dither.NONE)

    var keyed = Python.list()
    for j in range(len(small)):
        var rgba = small[j].quantize(palette=pal, dither=pil.Dither.NONE).convert("RGBA")
        rgba.putalpha(alphas[j])
        keyed.append(rgba)

    # 5. Drop empty frames, crop to the shared bounding box.
    var x0 = 1 << 30
    var y0 = 1 << 30
    var x1 = -1
    var y1 = -1
    var kept = Python.list()
    for f in keyed:
        var bbox = f.getbbox()
        if bbox is None:
            continue
        kept.append(f)
        x0 = min(x0, Int(py=bbox[0]))
        y0 = min(y0, Int(py=bbox[1]))
        x1 = max(x1, Int(py=bbox[2]))
        y1 = max(y1, Int(py=bbox[3]))
    if len(kept) == 0:
        raise Error("every frame was keyed out as background; try a lower --tol or check --bg")
    var fw = x1 - x0
    var fh = y1 - y0

    # 6. Write frames, strip and SpriteFrames resource.
    var frames_dir = out + "/" + name + "_frames"
    os.makedirs(frames_dir, exist_ok=True)
    var n = len(kept)
    var strip = pil.new("RGBA", Python.tuple(fw * n, fh), Python.tuple(0, 0, 0, 0))
    var k = 0
    for f in kept:
        var c = f.crop(Python.tuple(x0, y0, x1, y1))
        c.save(frames_dir + "/" + String(PythonObject(k).__str__().zfill(3)) + ".png")
        strip.paste(c, Python.tuple(k * fw, 0))
        k += 1
    var strip_name = name + ".png"
    strip.save(out + "/" + strip_name)

    var tres = String('[gd_resource type="SpriteFrames" load_steps=')
    tres += String(n + 2) + " format=3]\n\n"
    tres += '[ext_resource type="Texture2D" path="' + res_dir + strip_name + '" id="1_strip"]\n\n'
    for j in range(n):
        tres += '[sub_resource type="AtlasTexture" id="AtlasTexture_' + String(j) + '"]\n'
        tres += 'atlas = ExtResource("1_strip")\n'
        tres += "region = Rect2(" + String(j * fw) + ", 0, " + String(fw) + ", " + String(fh) + ")\n\n"
    tres += "[resource]\nanimations = [{\n\"frames\": ["
    for j in range(n):
        if j > 0:
            tres += ", "
        tres += '{\n"duration": 1.0,\n"texture": SubResource("AtlasTexture_' + String(j) + '")\n}'
    tres += '],\n"loop": true,\n"name": &"' + name + '",\n"speed": ' + String(fps) + ".0\n}]\n"
    Python.evaluate("lambda p, s: open(p, 'w').write(s)")(out + "/" + name + ".tres", tres)

    print("Wrote", n, "frames of", fw, "x", fh, "px to", out, "(" + strip_name + ",", name + ".tres)")
