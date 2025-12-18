# ----------------------------
# Very Simple Neural Network
# ----------------------------

# Input
x = 2.0

# True label (+1 or -1)
y = 1   # positive class

# Weight and bias (initial values)
w = 0.5
b = 0.1

# ----------------------------
# Forward Pass
# ----------------------------
y_pred = w * x + b
print("Predicted output:", y_pred)

# ----------------------------
# Hinge Loss
# ----------------------------
hinge_loss = max(0, 1 - y * y_pred)
print("Hinge Loss:", hinge_loss)
