import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# Initial parameters
w31, w32, w41, w42 = 0.3, 0.4, 0.2, 0.1
w53, w54 = 0.6, 0.5
b3, b4, b5 = -0.2, -0.1, -0.3
eta = 0.1

# AND training data
data = [(0,0,0), (0,1,0), (1,0,0), (1,1,1)]

# Part (a): Forward pass
print("=" * 60)
print("PART (a): Forward Pass")
print("=" * 60)

results = []
for x1, x2, t in data:
    z3 = w31*x1 + w32*x2 + b3
    z4 = w41*x1 + w42*x2 + b4
    o3 = sigmoid(z3)
    o4 = sigmoid(z4)
    z5 = w53*o3 + w54*o4 + b5
    o5 = sigmoid(z5)
    results.append((x1, x2, t, z3, z4, o3, o4, z5, o5))
    print(f"\nInput ({x1},{x2}), t={t}:")
    print(f"  z3 = {w31}*{x1} + {w32}*{x2} + ({b3}) = {z3:.3f}")
    print(f"  z4 = {w41}*{x1} + {w42}*{x2} + ({b4}) = {z4:.3f}")
    print(f"  o3 = sigmoid({z3:.3f}) = {o3:.3f}")
    print(f"  o4 = sigmoid({z4:.3f}) = {o4:.3f}")
    print(f"  z5 = {w53}*{o3:.3f} + {w54}*{o4:.3f} + ({b5}) = {z5:.3f}")
    print(f"  o5 = sigmoid({z5:.3f}) = {o5:.3f}")

# Part (b): BCE Loss
print("\n" + "=" * 60)
print("PART (b): BCE Loss and Backpropagation")
print("=" * 60)

losses = []
for x1, x2, t, z3, z4, o3, o4, z5, o5 in results:
    if t == 1:
        E = -math.log(o5)
    else:
        E = -math.log(1 - o5)
    losses.append(E)
    print(f"  E({x1},{x2}) = {E:.3f}")

E_total = sum(losses)
print(f"\n  Total batch loss E = {E_total:.3f}")
print(f"  E >= 0.1? {E_total >= 0.1} => Perform backpropagation")

# Output deltas
print("\n--- Output deltas (delta_5 = o5 - t) ---")
deltas_5 = []
for x1, x2, t, z3, z4, o3, o4, z5, o5 in results:
    d5 = o5 - t
    deltas_5.append(d5)
    print(f"  delta_5({x1},{x2}) = {o5:.3f} - {t} = {d5:.3f}")

# Output layer gradients
print("\n--- Output layer gradients ---")
n = len(data)
dE_dw53 = sum(deltas_5[i] * results[i][5] for i in range(n)) / n
dE_dw54 = sum(deltas_5[i] * results[i][6] for i in range(n)) / n
dE_db5 = sum(deltas_5[i] * 1 for i in range(n)) / n

print(f"  dE/dw53 = (1/{n}) * sum(delta_5 * o3) = {dE_dw53:.3f}")
for i in range(n):
    x1, x2, t = data[i]
    print(f"    ({x1},{x2}): {deltas_5[i]:.3f} * {results[i][5]:.3f} = {deltas_5[i]*results[i][5]:.3f}")

print(f"  dE/dw54 = (1/{n}) * sum(delta_5 * o4) = {dE_dw54:.3f}")
for i in range(n):
    x1, x2, t = data[i]
    print(f"    ({x1},{x2}): {deltas_5[i]:.3f} * {results[i][6]:.3f} = {deltas_5[i]*results[i][6]:.3f}")

print(f"  dE/db5 = (1/{n}) * sum(delta_5) = {dE_db5:.3f}")

# Hidden deltas
print("\n--- Hidden deltas ---")
deltas_3 = []
deltas_4 = []
for i, (x1, x2, t, z3, z4, o3, o4, z5, o5) in enumerate(results):
    d3 = deltas_5[i] * w53 * o3 * (1 - o3)
    d4 = deltas_5[i] * w54 * o4 * (1 - o4)
    deltas_3.append(d3)
    deltas_4.append(d4)
    print(f"  delta_3({x1},{x2}) = {deltas_5[i]:.3f} * {w53} * {o3:.3f} * (1-{o3:.3f}) = {d3:.3f}")
    print(f"  delta_4({x1},{x2}) = {deltas_5[i]:.3f} * {w54} * {o4:.3f} * (1-{o4:.3f}) = {d4:.3f}")

# Hidden layer gradients
print("\n--- Hidden layer gradients ---")
dE_dw31 = sum(deltas_3[i] * data[i][0] for i in range(n)) / n
dE_dw32 = sum(deltas_3[i] * data[i][1] for i in range(n)) / n
dE_dw41 = sum(deltas_4[i] * data[i][0] for i in range(n)) / n
dE_dw42 = sum(deltas_4[i] * data[i][1] for i in range(n)) / n
dE_db3 = sum(deltas_3[i] for i in range(n)) / n
dE_db4 = sum(deltas_4[i] for i in range(n)) / n

print(f"  dE/dw31 = (1/{n}) * sum(delta_3 * x1) = {dE_dw31:.3f}")
print(f"  dE/dw32 = (1/{n}) * sum(delta_3 * x2) = {dE_dw32:.3f}")
print(f"  dE/db3  = (1/{n}) * sum(delta_3)      = {dE_db3:.3f}")
print(f"  dE/dw41 = (1/{n}) * sum(delta_4 * x1) = {dE_dw41:.3f}")
print(f"  dE/dw42 = (1/{n}) * sum(delta_4 * x2) = {dE_dw42:.3f}")
print(f"  dE/db4  = (1/{n}) * sum(delta_4)      = {dE_db4:.3f}")

# Weight updates
print("\n--- Weight updates ---")
w53_new = w53 - eta * dE_dw53
w54_new = w54 - eta * dE_dw54
b5_new = b5 - eta * dE_db5
w31_new = w31 - eta * dE_dw31
w32_new = w32 - eta * dE_dw32
w41_new = w41 - eta * dE_dw41
w42_new = w42 - eta * dE_dw42
b3_new = b3 - eta * dE_db3
b4_new = b4 - eta * dE_db4

print(f"  w53 = {w53} - {eta}*{dE_dw53:.3f} = {w53_new:.3f}")
print(f"  w54 = {w54} - {eta}*{dE_dw54:.3f} = {w54_new:.3f}")
print(f"  b5  = {b5} - {eta}*{dE_db5:.3f} = {b5_new:.3f}")
print(f"  w31 = {w31} - {eta}*{dE_dw31:.3f} = {w31_new:.3f}")
print(f"  w32 = {w32} - {eta}*{dE_dw32:.3f} = {w32_new:.3f}")
print(f"  w41 = {w41} - {eta}*{dE_dw41:.3f} = {w41_new:.3f}")
print(f"  w42 = {w42} - {eta}*{dE_dw42:.3f} = {w42_new:.3f}")
print(f"  b3  = {b3} - {eta}*{dE_db3:.3f} = {b3_new:.3f}")
print(f"  b4  = {b4} - {eta}*{dE_db4:.3f} = {b4_new:.3f}")
