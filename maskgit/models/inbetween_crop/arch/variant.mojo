# In-betweening that only runs the tokens near the sprite: the target frame lies between the two
# keyframes, so the tokens within CROP of either keyframe's opaque pixels hold nearly all of it
# (99.8% of opaque target tokens at CROP = 3 on duelyst_hr). Every other token is transparent.
# Batches are grouped by that token count, so a typical batch runs about half the tokens.
comptime ARCH = "inbetween_crop"  # written into checkpoints; must match the folder name in models/
comptime CF = 2  # context frames on the canvas before the target: keyframe A, keyframe B
comptime STACKED = False  # context frames as extra tokens instead of embedding slots
comptime INBETWEEN = True  # dataset frame order: keyframe triples
comptime BAND = 16  # attention band in 64-token blocks; 16 spans a frame: full attention
comptime DYT = False  # DyT (Dynamic Tanh) in place of layer norm
comptime HILBERT = False  # tokens in Hilbert-curve order instead of row order
comptime CROP = 3  # >= 0: run only the target tokens within CROP tokens of the context sprites (see token_order in src/model.mojo)
