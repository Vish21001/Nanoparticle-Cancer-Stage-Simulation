import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Simulate cancer stages and treatments
# ----------------------------

np.random.seed(42)  # For reproducibility

stages = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]
n_samples = 1000  # Total simulated patients

# Simulate "true stages" randomly
true_stages = np.random.choice(stages, n_samples, p=[0.3,0.3,0.25,0.15])

# Simulate model predictions with some noise (generalization)
# High accuracy ~95%
accuracy_target = 0.95
pred_stages = []

for stage in true_stages:
    if np.random.rand() < accuracy_target:
        pred_stages.append(stage)  # Correct prediction
    else:
        # Choose an incorrect stage
        other_stages = [s for s in stages if s != stage]
        pred_stages.append(np.random.choice(other_stages))

# ----------------------------
# Compute validation accuracy
# ----------------------------
correct = sum([t==p for t,p in zip(true_stages, pred_stages)])
val_accuracy = correct / n_samples
print(f"Simulated Validation Accuracy: {val_accuracy*100:.2f}%")

# ----------------------------
# Prepare a chart (bar chart) for stage-wise performance
# ----------------------------
stage_counts = {stage:0 for stage in stages}
stage_correct = {stage:0 for stage in stages}

for t,p in zip(true_stages, pred_stages):
    stage_counts[t] += 1
    if t==p:
        stage_correct[t] += 1

# Accuracy per stage
stage_acc = {stage: stage_correct[stage]/stage_counts[stage]*100 for stage in stages}

# ----------------------------
# Plot as a chart (bar chart)
# ----------------------------
fig, ax = plt.subplots(figsize=(8,6))
bars = ax.bar(stage_acc.keys(), stage_acc.values(), color=["skyblue","orange","green","red"])
ax.set_ylim(0, 100)
ax.set_ylabel("Validation Accuracy (%)")
ax.set_title("Stage-wise Nanoparticle Model Validation Accuracy")

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height+1, f'{height:.1f}%', ha='center', va='bottom')

plt.show()
