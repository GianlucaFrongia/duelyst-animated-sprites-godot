# inbetween with band-sparse attention: each 64-token block (2 token rows, 4 pixel rows)
# attends only to the blocks at most BAND away. Everything else is models/inbetween (linked).
comptime ARCH = "inbetween_sparse"  # written into checkpoints; must match the folder name in models/
comptime BAND = 2  # 5 of the 16 blocks: about a third of full attention's work
comptime DYT = False  # DyT (Dynamic Tanh) in place of layer norm: models/inbetween_dyt
comptime HILBERT = False  # tokens in Hilbert-curve order instead of row order: models/inbetween_hilbert
