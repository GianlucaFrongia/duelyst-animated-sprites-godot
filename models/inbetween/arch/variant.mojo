# What sets this folder apart from the variants that link its other files (models/inbetween_sparse).
comptime ARCH = "inbetween"  # written into checkpoints; must match the folder name in models/
comptime BAND = 16  # attention band in 64-token blocks (src/metal.mojo); 16 spans the frame: full attention
comptime DYT = False  # DyT (Dynamic Tanh) in place of layer norm: models/inbetween_dyt
comptime HILBERT = False  # tokens in Hilbert-curve order instead of row order: models/inbetween_hilbert
