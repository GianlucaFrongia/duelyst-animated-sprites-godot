# inbetween with band-sparse attention: each 64-token block (2 token rows, 4 pixel rows)
# attends only to the blocks at most BAND away.
comptime ARCH = "inbetween_sparse"  # written into checkpoints; must match the folder name in models/
comptime CF = 2  # context frames on the canvas before the target: keyframe A, keyframe B
comptime STACKED = False  # context frames as extra tokens instead of embedding slots
comptime INBETWEEN = True  # dataset frame order: keyframe triples
comptime BAND = 2  # 5 of the 16 blocks: about a third of full attention's work
comptime DYT = False  # DyT (Dynamic Tanh) in place of layer norm
comptime HILBERT = False  # tokens in Hilbert-curve order instead of row order
comptime CROP = -1  # >= 0: run only the target tokens within CROP tokens of the context sprites (see token_order in src/model.mojo)
