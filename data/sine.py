import math

n = 628
with open('sine.csv', 'w') as f:
    f.write('t,sine_value\n')
    for t in range(n):
        x = t / 100
        f.write(f'{x},{math.sin(x):.4f}\n')
