# Train a LoRA on the exported Duelyst dataset with musubi-tuner
# (https://github.com/kohya-ss/musubi-tuner), driven from Mojo.
#
# PyTorch is used through Mojo's Python interop to check the GPU (ROCm or
# CUDA); the training itself is musubi-tuner's PyTorch code, launched as
# subprocesses: cache latents -> cache text encoder outputs -> train.
#
# Wan 2.1 1.3B (video clips):
#   mojo run lora/train_lora.mojo --model wan --musubi ~/musubi-tuner \
#       --dataset lora/dataset --dit models/wan2.1_t2v_1.3B_bf16.safetensors \
#       --vae models/wan_2.1_vae.safetensors \
#       --t5 models/models_t5_umt5-xxl-enc-bf16.pth
#
# FLUX.2 [klein] base 4B (spritesheet grids):
#   mojo run lora/train_lora.mojo --model flux2 --musubi ~/musubi-tuner \
#       --dataset lora/dataset --dit models/flux2-klein-base-4b.safetensors \
#       --vae models/ae.safetensors \
#       --text-encoder models/qwen3-4b/model-00001-of-00002.safetensors
#
# Add --dry-run to only print the commands and write the dataset config.

from std.python import Python, PythonObject
from std.sys import argv
from std.collections import List


def quote(s: String) raises -> String:
    return String(Python.import_module("shlex").quote(s))


def show(cmd: List[String]) raises -> String:
    var parts = List[String]()
    for c in cmd:
        parts.append(quote(c))
    return String(" ").join(parts)


def check_torch() raises:
    """Report the PyTorch build and GPU. ROCm builds expose AMD GPUs through
    the torch.cuda API, so the same calls work for AMD and NVIDIA."""
    var torch: PythonObject
    try:
        torch = Python.import_module("torch")
    except:
        print("PyTorch is not installed in the Python environment Mojo uses.")
        print("AMD:    pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4")
        print("NVIDIA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
        print("(check pytorch.org for the current ROCm/CUDA version)")
        raise Error("torch not found")
    var hip = torch.version.hip
    var backend = String("ROCm ") + String(hip) if hip is not None else String("CUDA ") + String(torch.version.cuda)
    print("PyTorch", String(torch.__version__), "|", backend)
    if not Bool(py=torch.cuda.is_available()):
        print("WARNING: no GPU visible to PyTorch. On RDNA2/RDNA3 cards that ROCm does not")
        print("officially list, try HSA_OVERRIDE_GFX_VERSION=10.3.0 (RDNA2) or 11.0.0 (RDNA3).")
        return
    var props = torch.cuda.get_device_properties(0)
    var gb = Float64(py=props.total_memory) / (1024.0 * 1024.0 * 1024.0)
    print("GPU:", String(props.name), "|", gb, "GB VRAM")
    if gb < 20.0:
        print("Note: <20 GB VRAM. Consider --fp8 and/or --blocks-to-swap N if you run out of memory.")


def main() raises:
    var model = String("wan")
    var musubi = String("")
    var dataset = String("lora/dataset")
    var work = String("lora/lora_work")
    var python = String("python")
    var dit = String("")
    var vae = String("")
    var t5 = String("")
    var text_encoder = String("")
    var task = String("t2v-1.3B")
    var model_version = String("klein-base-4b")
    var resolution = 0
    var max_frames = 33
    var rank = 32
    var lr = String("")
    var epochs = 16
    var name = String("")
    var optimizer = String("adamw")
    var fp8 = False
    var blocks_to_swap = 0
    var skip_cache = False
    var dry_run = False

    var args = argv()
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--dry-run":
            dry_run = True
        elif a == "--fp8":
            fp8 = True
        elif a == "--skip-cache":
            skip_cache = True
        else:
            if i + 1 >= len(args):
                raise Error("missing value for " + a)
            i += 1
            var v = String(args[i])
            if a == "--model": model = v
            elif a == "--musubi": musubi = v
            elif a == "--dataset": dataset = v
            elif a == "--work": work = v
            elif a == "--python": python = v
            elif a == "--dit": dit = v
            elif a == "--vae": vae = v
            elif a == "--t5": t5 = v
            elif a == "--text-encoder": text_encoder = v
            elif a == "--task": task = v
            elif a == "--model-version": model_version = v
            elif a == "--resolution": resolution = Int(v)
            elif a == "--max-frames": max_frames = Int(v)
            elif a == "--rank": rank = Int(v)
            elif a == "--lr": lr = v
            elif a == "--epochs": epochs = Int(v)
            elif a == "--name": name = v
            elif a == "--optimizer": optimizer = v
            elif a == "--blocks-to-swap": blocks_to_swap = Int(v)
            else:
                raise Error("unknown argument: " + a)
        i += 1

    if model != "wan" and model != "flux2":
        raise Error("--model must be 'wan' or 'flux2'")
    if musubi == "":
        raise Error("--musubi PATH (a musubi-tuner checkout) is required")

    var os = Python.import_module("os")
    var path = os.path
    var dataset_abs = String(path.abspath(dataset))
    var work_abs = String(path.abspath(work))
    if resolution == 0:
        resolution = 512 if model == "wan" else 1024
    if lr == "":
        lr = "2e-4" if model == "wan" else "1e-4"
    if name == "":
        name = "duelyst_" + model

    var missing = List[String]()
    if dit == "": missing.append("--dit")
    if vae == "": missing.append("--vae")
    if model == "wan" and t5 == "": missing.append("--t5")
    if model == "flux2" and text_encoder == "": missing.append("--text-encoder")
    if len(missing) > 0 and not dry_run:
        raise Error("missing model paths: " + String(", ").join(missing))
    # Model paths are resolved before we change into the musubi-tuner directory.
    if dit != "": dit = String(path.abspath(dit))
    if vae != "": vae = String(path.abspath(vae))
    if t5 != "": t5 = String(path.abspath(t5))
    if text_encoder != "": text_encoder = String(path.abspath(text_encoder))

    print("== Environment")
    if dry_run:
        try:
            check_torch()
        except:
            print("(continuing because of --dry-run)")
    else:
        check_torch()

    # ---- dataset config (musubi-tuner TOML) ----
    os.makedirs(work_abs, exist_ok=True)
    var toml = String("[general]\n")
    toml += "resolution = [" + String(resolution) + ", " + String(resolution) + "]\n"
    toml += 'caption_extension = ".txt"\nbatch_size = 1\nenable_bucket = true\n'
    # Never upscale the pixel art in the trainer (it would blur the pixel grid).
    toml += "bucket_no_upscale = true\n\n"
    toml += "[[datasets]]\n"
    if model == "wan":
        toml += 'video_directory = "' + dataset_abs + '/wan/videos"\n'
        toml += 'cache_directory = "' + work_abs + '/cache"\n'
        # Use each whole clip, trimmed to Wan's 4n+1 frame counts.
        toml += 'frame_extraction = "full"\n'
        toml += "max_frames = " + String(max_frames) + "\n"
    else:
        toml += 'image_directory = "' + dataset_abs + '/flux/grids"\n'
        toml += 'cache_directory = "' + work_abs + '/cache"\n'
    var toml_path = work_abs + "/dataset_" + model + ".toml"
    Python.evaluate("lambda p, s: open(p, 'w').write(s)")(toml_path, toml)
    print("\n== Dataset config written to", toml_path)
    print(toml)

    # ---- commands ----
    var cmds = List[List[String]]()
    var src = String("src/musubi_tuner/")
    if model == "wan":
        if not skip_cache:
            var c1: List[String] = [python, src + "wan_cache_latents.py",
                "--dataset_config", toml_path, "--vae", vae]
            cmds.append(c1^)
            var c2: List[String] = [python, src + "wan_cache_text_encoder_outputs.py",
                "--dataset_config", toml_path, "--t5", t5, "--batch_size", "16"]
            cmds.append(c2^)
        var t: List[String] = ["accelerate", "launch", "--num_cpu_threads_per_process", "1",
            "--mixed_precision", "bf16", src + "wan_train_network.py",
            "--task", task, "--dit", dit,
            "--timestep_sampling", "shift", "--discrete_flow_shift", "3.0",
            "--network_module", "networks.lora_wan"]
        cmds.append(t^)
    else:
        if not skip_cache:
            var c1: List[String] = [python, src + "flux_2_cache_latents.py",
                "--dataset_config", toml_path, "--vae", vae, "--model_version", model_version]
            cmds.append(c1^)
            var c2: List[String] = [python, src + "flux_2_cache_text_encoder_outputs.py",
                "--dataset_config", toml_path, "--text_encoder", text_encoder,
                "--batch_size", "16", "--model_version", model_version]
            cmds.append(c2^)
        var t: List[String] = ["accelerate", "launch", "--num_cpu_threads_per_process", "1",
            "--mixed_precision", "bf16", src + "flux_2_train_network.py",
            "--model_version", model_version, "--dit", dit, "--vae", vae,
            "--text_encoder", text_encoder,
            "--timestep_sampling", "flux2_shift", "--weighting_scheme", "none",
            "--network_module", "networks.lora_flux_2"]
        cmds.append(t^)

    # Options shared by both trainers. SDPA attention and plain AdamW are used by
    # default because xformers / bitsandbytes are unreliable on AMD ROCm.
    var common: List[String] = ["--dataset_config", toml_path, "--sdpa",
        "--mixed_precision", "bf16", "--optimizer_type", optimizer,
        "--learning_rate", lr, "--gradient_checkpointing",
        "--max_data_loader_n_workers", "2", "--persistent_data_loader_workers",
        "--network_dim", String(rank), "--max_train_epochs", String(epochs),
        "--save_every_n_epochs", "1", "--seed", "42",
        "--output_dir", work_abs + "/output", "--output_name", name]
    var last = len(cmds) - 1
    for c in common:
        cmds[last].append(c)
    if fp8:
        cmds[last].append("--fp8_base")
    if blocks_to_swap > 0:
        cmds[last].append("--blocks_to_swap")
        cmds[last].append(String(blocks_to_swap))

    var subprocess = Python.import_module("subprocess")
    for k in range(len(cmds)):
        print("\n== Step", k + 1, "of", len(cmds))
        print("(cd", quote(musubi), "&&", show(cmds[k]) + ")")
        if dry_run:
            continue
        var pyargs = Python.list()
        for c in cmds[k]:
            pyargs.append(c)
        var rc = Int(py=subprocess.run(pyargs, cwd=musubi).returncode)
        if rc != 0:
            raise Error("step " + String(k + 1) + " failed with exit code " + String(rc))

    if dry_run:
        print("\nDry run only: nothing was executed.")
    else:
        print("\nDone. LoRA files are in", work_abs + "/output")
