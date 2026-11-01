# The previous frame as 1,024 extra tokens: 2,048 tokens per canvas, and only the target
# frame's tokens are masked and predicted. The position embedding tells the frames apart.
comptime ARCH = "stacked"  # written into checkpoints; must match the folder name in models/
comptime CF = 1  # context frames on the canvas before the target
comptime STACKED = True  # context frames as extra tokens instead of embedding slots
comptime INBETWEEN = False  # dataset frame order: (previous..., next), not keyframe triples
comptime BAND = 16  # attention band in 64-token blocks; 16 spans a frame: full attention
comptime DYT = False  # DyT (Dynamic Tanh) in place of layer norm
comptime HILBERT = False  # tokens in Hilbert-curve order instead of row order
comptime CROP = -1  # >= 0: run only the target tokens within CROP tokens of the context sprites (see token_order in src/model.mojo)
