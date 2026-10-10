import os
import glob
import random

SOURCE_SPLIT_DIR = "configs/v1_irdt"
TARGET_SPLIT_DIR = "configs/v1_irdt_hardneg"
HARDNEG_DIR = r"C:\XboxGames\Dataset\speech_commands_v0.02\hard_negatives"

def main():
    os.makedirs(TARGET_SPLIT_DIR, exist_ok=True)
    
    # 1. Read existing v1_irdt splits
    with open(os.path.join(SOURCE_SPLIT_DIR, "train.txt"), "r") as f:
        train_lines = [l.strip() for l in f if l.strip()]
    with open(os.path.join(SOURCE_SPLIT_DIR, "valid.txt"), "r") as f:
        valid_lines = [l.strip() for l in f if l.strip()]
    with open(os.path.join(SOURCE_SPLIT_DIR, "test.txt"), "r") as f:
        test_lines = [l.strip() for l in f if l.strip()]

    print(f"Original v1_irdt: train={len(train_lines)}, valid={len(valid_lines)}, test={len(test_lines)}")

    # 2. Get all hard-negative wav files
    hardneg_files = sorted(glob.glob(os.path.join(HARDNEG_DIR, "*.wav")))
    print(f"Total hard-negative clips found: {len(hardneg_files)}")
    assert len(hardneg_files) > 0, "No hard-negative files found!"

    # Create relative paths with forward slash
    hardneg_rel = [f"hard_negatives/{os.path.basename(p)}" for p in hardneg_files]

    # Deterministic shuffle with fixed seed
    rng = random.Random(42)
    rng.shuffle(hardneg_rel)

    # Split: 50 test, 50 valid, rest (362) train
    foils_test = hardneg_rel[:50]
    foils_valid = hardneg_rel[50:100]
    foils_train = hardneg_rel[100:]

    new_train = train_lines + foils_train
    new_valid = valid_lines + foils_valid
    new_test = test_lines + foils_test

    # Shuffle training set so hard negatives are uniformly distributed
    rng.shuffle(new_train)

    # 3. Write target splits
    with open(os.path.join(TARGET_SPLIT_DIR, "train.txt"), "w") as f:
        f.write("\n".join(new_train) + "\n")
    with open(os.path.join(TARGET_SPLIT_DIR, "valid.txt"), "w") as f:
        f.write("\n".join(new_valid) + "\n")
    with open(os.path.join(TARGET_SPLIT_DIR, "test.txt"), "w") as f:
        f.write("\n".join(new_test) + "\n")

    # Also save dedicated test foils manifest for pure foil auditing
    with open(os.path.join(TARGET_SPLIT_DIR, "test_foils.txt"), "w") as f:
        f.write("\n".join(foils_test) + "\n")
    with open(os.path.join(TARGET_SPLIT_DIR, "all_foils.txt"), "w") as f:
        f.write("\n".join(hardneg_rel) + "\n")

    print(f"New v1_irdt_hardneg created successfully in {TARGET_SPLIT_DIR}:")
    print(f"  train.txt: {len(new_train)} ({len(train_lines)} original + {len(foils_train)} hard-negatives)")
    print(f"  valid.txt: {len(new_valid)} ({len(valid_lines)} original + {len(foils_valid)} hard-negatives)")
    print(f"  test.txt:  {len(new_test)} ({len(test_lines)} original + {len(foils_test)} hard-negatives)")
    print(f"  test_foils.txt: {len(foils_test)} held-out foil clips")
    print(f"  all_foils.txt:  {len(hardneg_rel)} total foil clips")

if __name__ == "__main__":
    main()
