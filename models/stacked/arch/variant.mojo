# What sets this folder apart from the variants that link its other files (models/sparse).
comptime ARCH = "stacked"  # written into checkpoints; must match the folder name in models/
comptime BAND = 16  # attention band in 64-token blocks (src/metal.mojo); 16 spans a frame: full attention
