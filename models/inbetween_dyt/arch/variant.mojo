# inbetween_sparse with every layer norm replaced by DyT, Dynamic Tanh (Zhu et al., "Transformers
# without Normalization", Meta 2025): y = g * tanh(a x) + b with one learned scalar a per norm.
comptime ARCH = "inbetween_dyt"  # written into checkpoints; must match the folder name in models/
comptime CF = 2  # context frames on the canvas before the target: keyframe A, keyframe B
comptime STACKED = False  # context frames as extra tokens instead of embedding slots
comptime INBETWEEN = True  # dataset frame order: keyframe triples
comptime BAND = 2  # 5 of the 16 blocks: about a third of full attention's work
comptime DYT = True  # DyT (Dynamic Tanh) in place of layer norm
comptime HILBERT = False  # tokens in Hilbert-curve order instead of row order
comptime CROP = -1  # >= 0: run only the target tokens within CROP tokens of the context sprites (see token_order in src/model.mojo)
