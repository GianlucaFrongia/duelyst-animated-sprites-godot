# inbetween_sparse with the 1024 tokens in Hilbert-curve order instead of row order (Li & Xu,
# "Hilbert-Guided Sparse Local Attention", ICLR 2026): each 64-token block is then an 8x8-token
# square (16x16 pixels) instead of 2 full-width token rows, so the band (BAND blocks either
# side along the curve) reaches about as far up and down as sideways. Same cost per step.
# Everything else is models/inbetween (linked).
comptime ARCH = "inbetween_hilbert"  # written into checkpoints; must match the folder name in models/
comptime BAND = 2
comptime DYT = False
comptime HILBERT = True
