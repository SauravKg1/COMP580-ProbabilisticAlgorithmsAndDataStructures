# Assignment 2 Report (2-universal hashing)

- Generated: 2025-10-18 14:54:15.955439
- Unique tokens: 445,165
- Approx dictionary memory: 48.20 MB
- Parameters: d=5, R=[1024, 16384, 262144]

## R=1024
- plots/errors_freq100_R1024_2u.png
- plots/errors_rand100_R1024_2u.png
- plots/errors_infreq100_R1024_2u.png

## R=16384
- plots/errors_freq100_R16384_2u.png
- plots/errors_rand100_R16384_2u.png
- plots/errors_infreq100_R16384_2u.png

## R=262144
- plots/errors_freq100_R262144_2u.png
- plots/errors_rand100_R262144_2u.png
- plots/errors_infreq100_R262144_2u.png

## Intersection vs R
- plots/intersection_vs_R_2u.png
- Intersections by sketch:
  - Count-Min: [100, 100, 100]
  - Count-Median: [72, 100, 100]
  - Count-Sketch: [90, 100, 100]

## Notes
- Count-Min overestimates; Count-Median reduces collision impact; Count-Sketch leverages ±1 signs.
- As R increases, collisions drop → lower relative errors and better heavy-hitter recovery.
- 2-universal hashing gives pairwise independence (a*x+b mod P); it is weaker than 4-universal but faster and often sufficient in practice.
