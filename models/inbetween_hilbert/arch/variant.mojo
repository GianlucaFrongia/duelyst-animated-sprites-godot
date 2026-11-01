# inbetween_sparse with the 1024 tokens in Hilbert-curve order instead of row order (Li & Xu,
# "Hilbert-Guided Sparse Local Attention", ICLR 2026): each 64-token block is then an 8x8-token
# square (16x16 pixels) instead of 2 full-width token rows, so the band (BAND blocks either
# side along the curve) reaches about as far up and down as sideways. Same cost per step.
comptime ARCH = "inbetween_hilbert"  # written into checkpoints; must match the folder name in models/
comptime CF = 2  # context frames on the canvas before the target: keyframe A, keyframe B
comptime STACKED = False  # context frames as extra tokens instead of embedding slots
comptime INBETWEEN = True  # dataset frame order: keyframe triples
comptime BAND = 2  # 5 of the 16 blocks: about a third of full attention's work
comptime DYT = False  # DyT (Dynamic Tanh) in place of layer norm
comptime HILBERT = True  # tokens in Hilbert-curve order instead of row order
comptime CROP = -1  # >= 0: run only the target tokens within CROP tokens of the context sprites (see token_order in src/model.mojo)
