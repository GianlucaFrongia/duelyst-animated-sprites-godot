# stacked with band-sparse attention: each 64-token block (2 token rows, 4 pixel rows)
# attends only to the blocks at most BAND away, in both frames.
comptime ARCH = "sparse"  # written into checkpoints; must match the folder name in models/
comptime CF = 1  # context frames on the canvas before the target
comptime STACKED = True  # context frames as extra tokens instead of embedding slots
comptime INBETWEEN = False  # dataset frame order: (previous..., next), not keyframe triples
comptime BAND = 2  # 5 of the 16 blocks per frame: about a third of full attention's work
comptime DYT = False  # DyT (Dynamic Tanh) in place of layer norm
comptime HILBERT = False  # tokens in Hilbert-curve order instead of row order
comptime CROP = -1  # >= 0: run only the target tokens within CROP tokens of the context sprites (see token_order in src/model.mojo)
