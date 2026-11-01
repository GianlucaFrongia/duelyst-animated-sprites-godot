# inbetween_dyt and inbetween_hilbert together: DyT in place of layer norm, and tokens in
# Hilbert-curve order so each attention block is a 16x16-pixel square.
comptime ARCH = "inbetween_dyt_hilbert"  # written into checkpoints; must match the folder name in models/
comptime CF = 2  # context frames on the canvas before the target: keyframe A, keyframe B
comptime STACKED = False  # context frames as extra tokens instead of embedding slots
comptime INBETWEEN = True  # dataset frame order: keyframe triples
comptime BAND = 2  # 5 of the 16 blocks: about a third of full attention's work
comptime DYT = True  # DyT (Dynamic Tanh) in place of layer norm
comptime HILBERT = True  # tokens in Hilbert-curve order instead of row order
comptime CROP = -1  # >= 0: run only the target tokens within CROP tokens of the context sprites (see token_order in src/model.mojo)
