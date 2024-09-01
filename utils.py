import audioflux as af
import numpy as np
from audioflux.type import SpectralFilterBankScaleType, SpectralDataType


def load_audio(path, samplerate=44100):
    """
    Load an audio file into an array using a specified sample rate.

    Parameters:
    - path (str): Path to the audio file.
    - samplerate (int): The sample rate to use for reading the audio. Default is 44100.

    Returns:
    - np.ndarray: The audio array.
    """
    audio_array, _ = af.read(path, samplate=samplerate)
    return audio_array


def trim_audio(audio_array, threshold=0.01):
    """
    Trim the silent parts from the beginning and end of an audio array.

    Parameters:
    - audio (np.ndarray): The audio array to be trimmed.
    - threshold (float): The threshold below which a sample is considered silent. Default is 0.01.

    Returns:
    - np.ndarray: The trimmed audio array.
    """
    non_silent_indices = np.where(np.abs(audio_array) > threshold)[0]
    if non_silent_indices.size > 0:
        start_index = non_silent_indices[0]
        end_index = non_silent_indices[-1]
        return audio_array[start_index:end_index + 1]
    # If the entire audio is silent, return the original array
    return audio_array


def get_cosine_similarity(vec1, vec2):
    """
    Compute the cosine similarity between two vectors.

    Parameters:
    - vec1 (np.ndarray): The first vector.
    - vec2 (np.ndarray): The second vector.

    Returns:
    - float: The cosine similarity between the two vectors.
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_pitch_similarity(audio_path, baseline_path, samplerate=44100):
    """
    Compare the pitch similarity between two audio files.

    Parameters:
    - audio_path (str): Path to the first audio file.
    - baseline_path (str): Path to the baseline audio file.
    - samplerate (int): Sampling rate to use for processing. Default is 44100.

    Returns:
    - float: Similarity percentage between the two audio files' pitches.
    """
    # Preprocess audio files
    audio = load_audio(audio_path)
    baseline = load_audio(baseline_path)
    audio = trim_audio(audio)
    baseline = trim_audio(baseline)

    # Extract pitch/frequency information
    pitch_obj = af.PitchYIN(samplate=samplerate)
    fre_arr_a, _, _ = pitch_obj.pitch(audio)
    fre_arr_b, _, _ = pitch_obj.pitch(baseline)

    # Interpolate if needed
    if fre_arr_a.shape != fre_arr_b.shape:
        min_len = min(fre_arr_a.shape[-1], fre_arr_b.shape[-1])
        if fre_arr_a.shape[-1] > fre_arr_b.shape[-1]:
            fre_arr_a = np.interp(np.linspace(0, 1, min_len), np.linspace(0, 1, fre_arr_a.shape[-1]), fre_arr_a)
        else:
            fre_arr_b = np.interp(np.linspace(0, 1, min_len), np.linspace(0, 1, fre_arr_b.shape[-1]), fre_arr_b)

    epsilon = 1e-8
    max_value = max(np.max(fre_arr_a), np.max(fre_arr_b), epsilon)
    a_normalized = fre_arr_a / max_value
    b_normalized = fre_arr_b / max_value

    total_diff = np.sum(np.abs(a_normalized - b_normalized))
    return 100 * (1 - total_diff / len(fre_arr_a))


def get_tone_similarity(audio_path, baseline_path, samplerate=44100):
    """
    Compare the tone similarity between two audio files using MFCC features.

    Parameters:
    - audio_path (str): Path to the first audio file.
    - baseline_path (str): Path to the baseline audio file.
    - samplerate (int): Sampling rate to use for processing. Default is 44100.

    Returns:
    - float: Similarity percentage between the two audio files' tones.
    """
    # Preprocess audio files
    audio = load_audio(audio_path)
    baseline = load_audio(baseline_path)
    audio = trim_audio(audio)
    baseline = trim_audio(baseline)

    # Extract MFCC features
    bft_obj = af.BFT(num=128, radix2_exp=12, samplate=samplerate,
                     scale_type=SpectralFilterBankScaleType.MEL,
                     data_type=SpectralDataType.POWER)
    spec_arr_a = bft_obj.bft(audio)
    spec_arr_a = np.abs(spec_arr_a)
    xxcc_obj_a = af.XXCC(bft_obj.num)
    xxcc_obj_a.set_time_length(time_length=spec_arr_a.shape[1])
    mfcc_arr_a = xxcc_obj_a.xxcc(spec_arr_a)

    spec_arr_b = bft_obj.bft(baseline)
    spec_arr_b = np.abs(spec_arr_b)
    xxcc_obj_b = af.XXCC(bft_obj.num)
    xxcc_obj_b.set_time_length(time_length=spec_arr_b.shape[1])
    mfcc_arr_b = xxcc_obj_b.xxcc(spec_arr_b)

    # Resize the longer array to match the shorter one
    min_length = min(mfcc_arr_a.shape[1], mfcc_arr_b.shape[1])
    if mfcc_arr_a.shape[1] > min_length:
        mfcc_arr_a = np.array(
            [np.interp(np.linspace(0, 1, min_length), np.linspace(0, 1, mfcc_arr_a.shape[1]), row) for row in
             mfcc_arr_a])
    elif mfcc_arr_b.shape[1] > min_length:
        mfcc_arr_b = np.array(
            [np.interp(np.linspace(0, 1, min_length), np.linspace(0, 1, mfcc_arr_b.shape[1]), row) for row in
             mfcc_arr_b])

    similarities = []
    for i in range(mfcc_arr_a.shape[1]):
        similarity = get_cosine_similarity(mfcc_arr_a[:, i], mfcc_arr_b[:, i])
        similarities.append(similarity)
    return ((np.mean(similarities) + 1) / 2) * 100


def get_loudness_similarity(audio_path, baseline_path):
    """
    Compare the loudness similarity between two audio files.

    Parameters:
    - audio_path (str): Path to the first audio file.
    - baseline_path (str): Path to the baseline audio file.

    Returns:
    - float: Similarity percentage between the two audio files' loudness.
    """
    # Preprocess audio files
    audio = load_audio(audio_path)
    baseline = load_audio(baseline_path)
    audio = trim_audio(audio)
    baseline = trim_audio(baseline)

    # Calculate the RMS (Root Mean Square) loudness for both audio arrays
    rms1 = np.sqrt(np.mean(np.square(audio)))
    rms2 = np.sqrt(np.mean(np.square(baseline)))

    # Normalize the loudness to avoid division by zero
    max_rms = max(rms1, rms2)
    min_rms = min(rms1, rms2)

    if max_rms == 0:
        return 100 if min_rms == 0 else 0

    return (min_rms / max_rms) * 100
