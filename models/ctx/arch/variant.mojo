# The previous frame as context, added into each target token's embedding: 1,024 tokens.
# Copying the last frame is the easy default, which keeps detail but moves little.
comptime ARCH = "ctx"  # written into checkpoints; must match the folder name in models/
comptime CF = 1  # context frames on the canvas before the target
comptime STACKED = False  # context frames as extra tokens instead of embedding slots
comptime INBETWEEN = False  # dataset frame order: (previous..., next), not keyframe triples
comptime BAND = 16  # attention band in 64-token blocks; 16 spans a frame: full attention
comptime DYT = False  # DyT (Dynamic Tanh) in place of layer norm
comptime HILBERT = False  # tokens in Hilbert-curve order instead of row order
comptime CROP = -1  # >= 0: run only the target tokens within CROP tokens of the context sprites (see token_order in src/model.mojo)
