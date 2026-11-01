# In-betweening: two keyframes (frames t and t+g) as context, the frame halfway between
# (t+g/2) as the target, so there is little ambiguity about where things move. A sheet is
# filled by halving gaps. Datasets: prepare_animations.py --inbetween.
comptime ARCH = "inbetween"  # written into checkpoints; must match the folder name in models/
comptime CF = 2  # context frames on the canvas before the target: keyframe A, keyframe B
comptime STACKED = False  # context frames as extra tokens instead of embedding slots
comptime INBETWEEN = True  # dataset frame order: keyframe triples
comptime BAND = 16  # attention band in 64-token blocks; 16 spans a frame: full attention
comptime DYT = False  # DyT (Dynamic Tanh) in place of layer norm
comptime HILBERT = False  # tokens in Hilbert-curve order instead of row order
comptime CROP = -1  # >= 0: run only the target tokens within CROP tokens of the context sprites (see token_order in src/model.mojo)
