# src/train.py
import os
import argparse
import zipfile
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks
import tensorflow as tf

from audio_utils import wav_to_logmel, pad_or_crop_mel, SR, N_MELS, FIXED_TIME_FRAMES

def extract_zip(zip_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(out_dir)

def load_dataset(makina_dir, newstyle_dir):
    X, y = [], []
    label_map = {"makina":0, "newstyle":1}

    for label_name, folder in [("makina", makina_dir), ("newstyle", newstyle_dir)]:
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".wav")])
        for fname in tqdm(files, desc=f"Cargando {label_name}"):
            fpath = os.path.join(folder, fname)
            mel = wav_to_logmel(fpath)
            mel = pad_or_crop_mel(mel, FIXED_TIME_FRAMES)
            X.append(mel)
            y.append(label_map[label_name])

    X = np.array(X)[..., np.newaxis]  # (N, 128, 431, 1)
    y = np.array(y)
    return X, y

def build_model(input_shape=(N_MELS, FIXED_TIME_FRAMES, 1), num_classes=2):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main(args):
    # preparar datos
    work_data = args.data_dir
    os.makedirs(work_data, exist_ok=True)

    makina_dir = os.path.join(work_data, "makina")
    newstyle_dir = os.path.join(work_data, "newstyle")
    os.makedirs(makina_dir, exist_ok=True)
    os.makedirs(newstyle_dir, exist_ok=True)

    # Si vienen ZIPs, los extraemos
    if args.makina_zip:
        extract_zip(args.makina_zip, makina_dir)
    if args.newstyle_zip:
        extract_zip(args.newstyle_zip, newstyle_dir)

    # Cargar dataset
    X, y = load_dataset(makina_dir, newstyle_dir)

    # Splits
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Modelo
    model = build_model()
    es = callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy")
    rlrop = callbacks.ReduceLROnPlateau(patience=2, factor=0.5)

    model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        callbacks=[es, rlrop],
        verbose=1
    )

    # Evaluación
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")

    # Guardar
    os.makedirs(args.models_dir, exist_ok=True)
    out_path = os.path.join(args.models_dir, args.model_name)
    model.save(out_path)
    print("Modelo guardado en:", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", help="Carpeta con /makina y /newstyle")
    parser.add_argument("--makina_zip", default=None, help="ZIP con WAVs de 10s de Makina")
    parser.add_argument("--newstyle_zip", default=None, help="ZIP con WAVs de 10s de Newstyle")
    parser.add_argument("--models_dir", default="models", help="Carpeta de salida del modelo")
    parser.add_argument("--model_name", default="fabrik_makina_newstyle.h5", help="Nombre del archivo del modelo")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    main(args)
