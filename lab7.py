# LAB7
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

print("Train:", x_train.shape, y_train.shape)
print("Test:",  x_test.shape,  y_test.shape)

def build_transfer_model(base="mobilenet", fine_tune=False):
    input_shape = (224, 224, 3)

    if base == "mobilenet":
        base_model = keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet"
        )
    elif base == "resnet50":
        base_model = keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet"
        )
    else:
        raise ValueError("base must be 'mobilenet' or 'resnet50'")

    base_model.trainable = fine_tune

    model = keras.Sequential([
        layers.Resizing(224, 224),
        layers.Rescaling(1./255),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

print("\n========== MobileNetV2 (Transfer Learning) ==========")
model_mobilenet = build_transfer_model(base="mobilenet", fine_tune=False)
model_mobilenet.summary()

model_mobilenet.fit(
    x_train, y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.2,
    verbose=2
)

loss_m, acc_m = model_mobilenet.evaluate(x_test, y_test, verbose=0)
print(f"MobileNetV2 Accuracy: {acc_m:.4f}")

print("\n=============== ResNet50 (Transfer Learning) ===============")
model_resnet = build_transfer_model(base="resnet50", fine_tune=False)
model_resnet.summary()

model_resnet.fit(
    x_train, y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.2,
    verbose=2
)

loss_r, acc_r = model_resnet.evaluate(x_test, y_test, verbose=0)
print(f"ResNet50 Accuracy: {acc_r:.4f}")

print("\n=============== ResNet50 (Fine-Tuning) ===============")
model_resnet_ft = build_transfer_model(base="resnet50", fine_tune=True)
model_resnet_ft.summary()

model_resnet_ft.fit(
    x_train, y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.2,
    verbose=2
)

loss_ft, acc_ft = model_resnet_ft.evaluate(x_test, y_test, verbose=0)
print(f"ResNet50 Fine-Tuned Accuracy: {acc_ft:.4f}")

print("\n=================== DONE ===================")
