import torch

def parametric_f0(midi_pitch, dur, vibrato_rate_hz=6.0, vibrato_depth_cent=60.0, sr=24000, hop=240):
    # Returns frame-level F0 curve (Hz) guided by MIDI + vibrato
    # This is a simplified stub.
    B, T = midi_pitch.shape
    frames_per_note = (dur * sr / hop).long().clamp(min=1)
    f0_tracks = []
    for b in range(B):
        f0 = []
        for t in range(T):
            n_frames = int(frames_per_note[b, t].item())
            hz = 440.0 * (2 ** ((midi_pitch[b, t].item() - 69)/12))
            # vibrato
            import numpy as np
            idx = np.arange(n_frames)
            vib = np.sin(2*np.pi*idx*(vibrato_rate_hz*(hop/sr))) * (vibrato_depth_cent/1200.0)
            hz_curve = hz * (2 ** vib)
            f0.extend(hz_curve.tolist())
        f0_tracks.append(torch.tensor(f0))
    return f0_tracks
