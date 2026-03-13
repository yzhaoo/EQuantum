import numpy as np

# Simulate parameters
lat_spacing = 0.005 # from spacing0
t = 0.4388 / (lat_spacing * 1000) # honeycomb t calculation
bandwidth = 2.9 * t

delta = t / 20.0
npol_scale = 6
scale = bandwidth # estimate for scale? Or maybe scale = estimate_bandwidth(m)? 
# In estimate_bandwidth, it finds max eigenvalue. It's usually close to bandwidth.
npol = int(npol_scale * scale / delta)
ne = npol * 10

print(f"t: {t}")
print(f"bandwidth: {bandwidth}")
print(f"delta: {delta}")
print(f"scale: {scale}")
print(f"npol: {npol}")
print(f"ne: {ne}")
