# The two previous frames as context, added into each target token's embedding: with two
# frames the model can see which way the sprite moves (one frame can't tell breathing in
# from breathing out). Datasets: prepare_animations.py --context 2.
comptime ARCH = "ctx2"  # written into checkpoints; must match the folder name in models/
comptime CF = 2  # context frames on the canvas before the target
comptime STACKED = False  # context frames as extra tokens instead of embedding slots
comptime INBETWEEN = False  # dataset frame order: (previous..., next), not keyframe triples
comptime BAND = 16  # attention band in 64-token blocks; 16 spans a frame: full attention
comptime DYT = False  # DyT (Dynamic Tanh) in place of layer norm
comptime HILBERT = False  # tokens in Hilbert-curve order instead of row order
comptime CROP = -1  # >= 0: run only the target tokens within CROP tokens of the context sprites (see token_order in src/model.mojo)
