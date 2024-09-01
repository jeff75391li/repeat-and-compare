import os
import tempfile
import audioflux as af
import streamlit as st
import matplotlib.pyplot as plt
from st_audiorec import st_audiorec
from utils import load_audio, trim_audio, get_pitch_similarity, get_tone_similarity, get_loudness_similarity


st.set_page_config(page_title="SpeakRepeat-Demo", page_icon="🗣️")
st.title("Listen, and then repeat!")
st.audio('reference.wav', format='audio/wav')

wav_audio_data = st_audiorec()

submitted = st.button("Let's valuate your speech")
if submitted:
    if wav_audio_data is None:
        st.error('Please record your speech!')
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        temp_file.write(wav_audio_data)
        temp_path = temp_file.name

    with st.spinner('AI is determining your similarity score...'):
        audio_arr = load_audio(temp_path)
        baseline_arr = load_audio('reference.wav')
        audio_arr = trim_audio(audio_arr)
        baseline_arr = trim_audio(baseline_arr)

        ps = get_pitch_similarity(temp_path, 'reference.wav')
        ts = get_tone_similarity(temp_path, 'reference.wav')
        ls = get_loudness_similarity(temp_path, 'reference.wav')
    st.success('Success!')
    st.write(f'Pitch similarity: {ps:.2f}%')
    st.write(f'Tone similarity: {ts:.2f}%')
    st.write(f'Loudness similarity: {ls:.2f}%')

    st.subheader('Audio waveforms:')
    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].set_title('Reference speech')
    af.display.fill_wave(baseline_arr, samplate=44100, axes=ax[0])
    ax[1].set_title('Your speech')
    af.display.fill_wave(audio_arr, samplate=44100, axes=ax[1])
    st.pyplot(fig)

    if temp_path is not None:
        os.remove(temp_path)
