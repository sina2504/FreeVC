import os
import argparse
import torch

import librosa
import time
from io import BytesIO
import soundfile as sf  # used for writing WAV data to a buffer
import logging
logging.getLogger('numba').setLevel(logging.WARNING)

import utils
from models import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
from wavlm import WavLM, WavLMConfig
from speaker_encoder.voice_encoder import SpeakerEncoder

def load_models(hpfile, ptfile):
    # Load hyperparameters and models
    hps = utils.get_hparams_from_file(hpfile)
    print("Loading model...")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).cuda()
    net_g.eval()
    print("Loading checkpoint...")
    utils.load_checkpoint(ptfile, net_g, None, True)

    print("Loading WavLM for content...")
    cmodel = utils.get_cmodel(0)

    if hps.model.use_spk:
        print("Loading speaker encoder...")
        smodel = SpeakerEncoder('speaker_encoder/ckpt/pretrained_bak_5805000.pt')
    else:
        smodel = None

    return hps, net_g, cmodel, smodel

def convert_audio_buffer(source_buffer: BytesIO, target_buffer: BytesIO, title: str,
                         hps, net_g, cmodel, smodel, use_timestamp=False) -> BytesIO:
    """
    Accepts in-memory source and target audio buffers, processes the conversion,
    and returns the output audio as a BytesIO buffer.
    """
    # Ensure buffers are at the beginning
    source_buffer.seek(0)
    target_buffer.seek(0)
    
    # Load and preprocess target audio
    wav_tgt, _ = librosa.load(target_buffer, sr=hps.data.sampling_rate)
    wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
    if hps.model.use_spk:
        g_tgt = smodel.embed_utterance(wav_tgt)
        g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).cuda()
    else:
        wav_tgt_tensor = torch.from_numpy(wav_tgt).unsqueeze(0).cuda()
        mel_tgt = mel_spectrogram_torch(
            wav_tgt_tensor, 
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax
        )

    # Load and preprocess source audio
    wav_src, _ = librosa.load(source_buffer, sr=hps.data.sampling_rate)
    wav_src_tensor = torch.from_numpy(wav_src).unsqueeze(0).cuda()
    c = utils.get_content(cmodel, wav_src_tensor)
    
    # Run inference
    if hps.model.use_spk:
        audio = net_g.infer(c, g=g_tgt)
    else:
        audio = net_g.infer(c, mel=mel_tgt)
    audio = audio[0][0].data.cpu().float().numpy()

    # Write output audio to an in-memory buffer using soundfile
    output_buffer = BytesIO()
    sf.write(output_buffer, audio, hps.data.sampling_rate, format='WAV')
    output_buffer.seek(0)
    return output_buffer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpfile", type=str, default="configs/freevc.json", help="Path to JSON config file")
    parser.add_argument("--ptfile", type=str, default="checkpoints/freevc.pth", help="Path to pth file")
    parser.add_argument("--use_timestamp", default=False, action="store_true", help="Append timestamp to output title")
    args = parser.parse_args()
    
    # Load the model and related components once
    hps, net_g, cmodel, smodel = load_models(args.hpfile, args.ptfile)
    
    # For testing, assume there is two WAV files in memory.
    # In practice, these buffers can be provided directly (e.g., via a web API).
    source_path = "example_source.wav"
    target_path = "example_target.wav"
    with open(source_path, "rb") as f:
        source_bytes = f.read()
    with open(target_path, "rb") as f:
        target_bytes = f.read()
    
    source_buffer = BytesIO(source_bytes)
    target_buffer = BytesIO(target_bytes)
    
    # Use a title (or unique identifier) for this conversion
    title = "sample"
    if args.use_timestamp:
        timestamp = time.strftime("%m-%d_%H-%M", time.localtime())
        title = f"{timestamp}_{title}"
    
    # Convert audio entirely in-memory
    output_buffer = convert_audio_buffer(source_buffer, target_buffer, title, hps, net_g, cmodel, smodel, use_timestamp=args.use_timestamp)
    
    # For demonstration, write the output buffer to disk
    with open("output_sample.wav", "wb") as f:
        f.write(output_buffer.read())
