import os
import json
import numpy as np

INPUT_FOLDER = "landmarks_dataset"
OUTPUT_FOLDER = "processed_landmarkers"
TARGET_LENGTH = 30

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def normalize_frame(frame):
    """
    Converts a frame into a normalized 126-value vector.

    Supported input formats:

    [[63],[63]]
    [[63]]
    [126]
    [63]
    """

    # -----------------------------
    # Convert to 126 values
    # -----------------------------
    if len(frame) == 0:
        frame = [0.0] * 126

    # Dataset format:
    # [
    #    [63 values],
    #    [63 values]
    # ]
    elif isinstance(frame[0], list):

        if len(frame) == 1:
            left = frame[0]
            right = [0.0] * 63

        else:
            left = frame[0]
            right = frame[1]

        frame = left + right

    # Already flattened 63 values
    elif len(frame) == 63:
        frame = frame + [0.0] * 63

    # Already flattened 126 values
    elif len(frame) == 126:
        pass

    else:
        raise ValueError(f"Unexpected frame length: {len(frame)}")

    # -----------------------------
    # Normalization
    # -----------------------------
    frame = np.array(frame, dtype=np.float32).reshape(-1, 3)

    # Translation normalization
    origin = frame[0]
    frame = frame - origin

    # Scale normalization
    scale = np.linalg.norm(frame, axis=1).max()

    if scale > 0:
        frame = frame / scale

    return frame.flatten().tolist()


def normalize_sequence(sequence):
    return [normalize_frame(frame) for frame in sequence]


def resample_sequence(sequence, target_len=30):

    if len(sequence) == 0:
        return []

    indices = np.linspace(
        0,
        len(sequence) - 1,
        target_len
    ).astype(int)

    return [sequence[i] for i in indices]


def process_file(filepath):

    with open(filepath, "r") as f:
        data = json.load(f)

    # Raw dataset
    if isinstance(data, list):
        sequence = data
        sign_name = os.path.splitext(os.path.basename(filepath))[0]

    # Already structured dataset
    else:
        sequence = data.get("sequence", [])
        sign_name = data.get(
            "sign",
            os.path.splitext(os.path.basename(filepath))[0]
        )

    # Normalize landmarks
    sequence = normalize_sequence(sequence)

    # Resample to 30 frames
    sequence = resample_sequence(sequence, TARGET_LENGTH)

    return {
        "sign": sign_name,
        "sequence": sequence
    }


def process_all():

    files = [
        f for f in os.listdir(INPUT_FOLDER)
        if f.endswith(".json")
    ]

    print(f"\nFound {len(files)} files\n")

    success = 0

    for file in files:

        input_path = os.path.join(INPUT_FOLDER, file)
        output_path = os.path.join(OUTPUT_FOLDER, file)

        try:

            processed = process_file(input_path)

            with open(output_path, "w") as f:
                json.dump(processed, f)

            success += 1

            print(
                f"✔ {processed['sign']}"
                f" | Frames: {len(processed['sequence'])}"
                f" | Frame size: {len(processed['sequence'][0])}"
            )

        except Exception as e:

            print(f"❌ {file}")
            print(e)

    print("\n==============================")
    print(f"Finished! {success}/{len(files)} processed.")
    print("==============================")


if __name__ == "__main__":
    process_all()