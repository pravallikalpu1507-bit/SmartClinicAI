import pandas as pd
from models.preprocess import load_data, clean_data
from models.structured_model import train_models
from models.visualization import plot_results

# Load & clean
data = load_data()
data = clean_data(data)

print("\nColumn names:", data.columns.tolist())

# Features & labels
X = data.drop("target", axis=1)
y = data["target"]

# Train all models
results, trained_models = train_models(X, y)

# Print results table
print("\n" + "="*40)
print("   MODEL ACCURACY RESULTS")
print("="*40)
for model_name, acc in results.items():
    bar = "█" * int(acc / 5)
    print(f"  {model_name:<15} {acc:>6.2f}%  {bar}")
print("="*40)

best = max(results, key=results.get)
print(f"\n🏆 Best model: {best} ({results[best]:.2f}%)")

# Plot
plot_results(results)
