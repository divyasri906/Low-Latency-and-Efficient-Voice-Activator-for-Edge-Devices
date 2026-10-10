"""
Generate Hard-Negative Confusable Audio Dataset for KWS Marvin Model.
Generates phonetically similar words (foils) to eliminate false alarms:
- bargain, martin, margin, garden, pardon, carbon, tartan, morgan
- market, marvel, marble, markup, morning, mining

Uses edge-tts with diverse multi-accent neural voices, rates, and pitches.
Saves 16 kHz 16-bit mono 1.0s WAV files into dataset directory.
"""

import os
import asyncio
import io
import wave
import numpy as np
import soundfile as sf
import edge_tts

OUTPUT_DIR = r"C:\XboxGames\Dataset\speech_commands_v0.02\hard_negatives"

# Confusable words targeting specific phonemic near-misses
FOIL_WORDS = [
    "bargain",
    "martin",
    "margin",
    "garden",
    "pardon",
    "carbon",
    "tartan",
    "morgan",
    "market",
    "marvel",
    "marble",
    "markup",
    "morning",
    "mining",
]

# Diverse voices across accents, genders, and acoustics
VOICES = [
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-US-AriaNeural",
    "en-US-ChristopherNeural",
    "en-US-EricNeural",
    "en-GB-RyanNeural",
    "en-GB-SoniaNeural",
    "en-IN-PrabhatNeural",
    "en-IN-NeerjaNeural",
    "en-AU-WilliamNeural",
    "en-AU-NatashaNeural",
]

RATES = ["-10%", "+0%", "+10%"]
PITCHES = ["-5Hz", "+0Hz", "+5Hz"]


def pad_or_crop_to_1sec(audio_data, sr=16000, target_samples=16000):
    """Ensure audio is exactly 1.0 second (16000 samples @ 16kHz) centered."""
    if len(audio_data) > target_samples:
        # Trim excess from ends
        start = (len(audio_data) - target_samples) // 2
        return audio_data[start : start + target_samples]
    elif len(audio_data) < target_samples:
        # Center-pad with silence
        pad_total = target_samples - len(audio_data)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(audio_data, (pad_left, pad_right), mode="constant")
    return audio_data


async def generate_single_sample(word, voice, rate, pitch, out_path):
    communicate = edge_tts.Communicate(word, voice, rate=rate, pitch=pitch)
    mp3_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_data += chunk["data"]

    if not mp3_data:
        raise ValueError(f"No audio received for {word} ({voice})")

    # Read mp3 into soundfile
    with io.BytesIO(mp3_data) as bio:
        data, sr = sf.read(bio)

    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Resample to 16 kHz if necessary
    if sr != 16000:
        import scipy.signal

        num_samples = int(round(len(data) * 16000 / sr))
        data = scipy.signal.resample(data, num_samples)

    # Normalize amplitude to standard Speech Commands volume range
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val * 0.85  # ~-1.5 dBFS

    # Center-pad or crop to exactly 1.0s (16,000 samples)
    data_1sec = pad_or_crop_to_1sec(data, sr=16000, target_samples=16000)
    int16_pcm = np.clip(data_1sec * 32767.0, -32768, 32767).astype(np.int16)

    # Write as standard WAV
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(int16_pcm.tobytes())


async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating hard-negative samples in: {OUTPUT_DIR}")

    tasks = []
    sample_idx = 0

    for word in FOIL_WORDS:
        for voice in VOICES:
            for rate in ["-10%", "+0%", "+10%"]:
                pitch = "+0Hz"
                filename = f"{word}_{voice.split('-')[2]}_{rate.replace('%','').replace('+','p').replace('-','m')}_{sample_idx:04d}.wav"
                out_path = os.path.join(OUTPUT_DIR, filename)
                sample_idx += 1
                tasks.append((word, voice, rate, pitch, out_path))

    print(f"Total target hard-negative clips to synthesize: {len(tasks)}")

    # Execute with bounded concurrency
    sem = asyncio.Semaphore(10)

    async def worker(t):
        word, voice, rate, pitch, out_path = t
        if os.path.exists(out_path):
            return
        async with sem:
            for attempt in range(3):
                try:
                    await generate_single_sample(word, voice, rate, pitch, out_path)
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"Failed {word} ({voice}): {e}")
                    await asyncio.sleep(1)

    batch_size = 50
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i : i + batch_size]
        await asyncio.gather(*[worker(t) for t in batch])
        print(f"  --> Synthesized {min(i + batch_size, len(tasks))} / {len(tasks)} samples...")

    # Count generated files
    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".wav")]
    print(f"\n[DONE] Successfully synthesized {len(files)} hard-negative audio clips in {OUTPUT_DIR}!\n")


if __name__ == "__main__":
    asyncio.run(main())
