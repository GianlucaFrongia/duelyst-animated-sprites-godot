# inbetween_sparse with every layer norm replaced by DyT, Dynamic Tanh (Zhu et al., "Transformers
# without Normalization", Meta 2025): y = g * tanh(a x) + b with one learned scalar a per norm.
# Everything else is models/inbetween (linked).
comptime ARCH = "inbetween_dyt"  # written into checkpoints; must match the folder name in models/
comptime BAND = 2
comptime DYT = True
comptime HILBERT = False  # tokens in Hilbert-curve order instead of row order: models/inbetween_hilbert
