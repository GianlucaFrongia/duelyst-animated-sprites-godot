# The stacked canvas with band-sparse attention: each 64-token block (2 token rows, 4 pixel
# rows) attends only to the blocks at most BAND away in both frames. Everything else is
# models/stacked (linked).
comptime ARCH = "sparse"  # written into checkpoints; must match the folder name in models/
comptime BAND = 2  # 5 of the 16 blocks per frame: about a third of full attention's work
