import argparse
from pathlib import Path
import numpy as np


def extract_vector(audio_path: Path) -> np.ndarray:
    from paddlespeech.cli.vector import VectorExecutor

    executor = VectorExecutor()
    vector = executor(audio_file=str(audio_path))
    return np.array(vector)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = vec_a.astype(np.float64)
    b = vec_b.astype(np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def main():
    parser = argparse.ArgumentParser(description="View and compare voiceprint vectors (speaker embeddings)")
    parser.add_argument("wav", nargs="+", help="One or more .wav files to extract vectors from")
    parser.add_argument("--save", action="store_true", help="Save vectors as .npy alongside the wav files")
    args = parser.parse_args()

    paths = [Path(p).expanduser().resolve() for p in args.wav]

    vectors = []
    for p in paths:
        if not p.exists():
            print(f"[ERROR] File not found: {p}")
            return
        vec = extract_vector(p)
        vectors.append(vec)
        print(f"\nFile: {p.name}")
        print(f"Shape: {vec.shape}, dtype: {vec.dtype}")
        print(f"First 10 values: {np.array2string(vec[:10], precision=6, separator=', ') }")
        if args.save:
            out = p.with_suffix(".npy")
            np.save(out, vec)
            print(f"Saved: {out}")

    if len(vectors) >= 2:
        print("\nPairwise cosine similarity:")
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                sim = cosine_similarity(vectors[i], vectors[j])
                print(f"  {paths[i].name} vs {paths[j].name}: {sim:.6f}")


if __name__ == "__main__":
    main()


