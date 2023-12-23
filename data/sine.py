import math

n = 1000
with open('sine.csv', 'w') as f:
    f.write('t,sine_value\n')
    for t in range(1, n + 1):
        x = t * 2 * math.pi / 1000
        f.write(f'{x},{math.sin(x):.4f}\n')
