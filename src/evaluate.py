from preprocessing import load_and_preprocess
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

X_train , X_val , X_test , y_train , y_val , y_test = load_and_preprocess()

model_path = Path(__file__).resolve().parent.parent / "artifacts" / "fashion_cnn.keras"

model = load_model(model_path)

test_loss , test_accuracy = model.evaluate(X_test , y_test)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

report_dir = Path(__file__).resolve().parent.parent / "reports"
report_dir.mkdir(parents=True, exist_ok=True)

metrics = {
    "test_loss" : float(test_loss),
    "test_accuracy" : float(test_accuracy),
}

with open(report_dir / "evaluation_metrics.json", "w") as file:
    json.dump(metrics , file ,indent=4)

predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions , axis =1)

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress",
    "Coat", "Sandal", "Shirt", "Sneaker",
    "Bag", "Ankle boot"
]

report = classification_report(
    y_test,
    predicted_labels,
    target_names=class_names
)

print(predictions.shape)
print(predicted_labels.shape)
print(y_test.shape)
print(report)

with open(report_dir / "classification_report.txt", "w") as file:
    file.write(report)

cm = confusion_matrix(y_test, predicted_labels)
#creates 10 x 10 matrix where row = class , column = predicted class

#visualising the matrix 

fig, ax = plt.subplots(figsize=(10, 8))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(ax=ax, cmap="Blues", xticks_rotation=45)

#saving the figure
plt.tight_layout()
plt.savefig(report_dir / "confusion_matrix.png", dpi=300)
plt.close(fig)

misclassified_indices = np.where(
    predicted_labels != y_test
)[0]

print("Total misclassified:", len(misclassified_indices))

rng = np.random.default_rng(42)

selected_indices = rng.choice(
    misclassified_indices,
    size=12,
    replace=False
)

fig, axes = plt.subplots(3, 4, figsize=(12, 9))
for ax, idx in zip(axes.flat, selected_indices):

    image = X_test[idx].squeeze()

    actual = class_names[y_test[idx]]
    predicted = class_names[predicted_labels[idx]]

    ax.imshow(image, cmap="gray", vmin=image.min(), vmax=image.max())

    ax.set_title(f"Actual: {actual}\nPredicted: {predicted}")
    ax.axis("off")

fig.suptitle("Misclassified Fashion-MNIST Images", fontsize=16)

plt.tight_layout()
plt.savefig(report_dir / "misclassified_examples.png", dpi=300)
plt.close(fig)    


