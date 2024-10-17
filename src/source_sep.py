import itertools
import random
import numpy as np
import os

import torch
from torch.utils.data import TensorDataset, DataLoader

import torchaudio
from torchaudio.pipelines import CONVTASNET_BASE_LIBRI2MIX

from torchmetrics.audio import ScaleInvariantSignalNoiseRatio, SignalDistortionRatio, PermutationInvariantTraining
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio, signal_distortion_ratio

from tqdm import tqdm

from train_test_split_list import train_artists, val_artists, test_artists


def combine_vocals(vocals1, vocals2, sample_rate, save_as=None):
    """
    Combine 2 vocal audio waves.

    Saves mixed vocals to path `save_as`
    Returns mixed vocal wave.
    """
    if vocals1.shape[0] > vocals2.shape[0]:
        shorter, longer = vocals2, vocals1
    else:
        shorter, longer = vocals1, vocals2
    
    combined_len = shorter.shape[0]
    # resize longer vocals to match shape of shorter vocals
    longer_trimmed = longer[:combined_len]

    # mix both vocals
    mixed_tensor = torch.cat([longer_trimmed.reshape(1, combined_len), shorter.reshape(1, combined_len)])
    mixed = torch.mean(mixed_tensor, dim=0, keepdim=True)
    
    if save_as:
        torchaudio.save(f'{save_as}', mixed, sample_rate)
        
    return mixed, mixed_tensor


def load_vocals(file_pair, target_sr=8000):
    """
    Loads a pair of audio files, resamples them to the target sample rate (target_sr), and extract vocals only
    """
    audio1, sample_rate = torchaudio.load(f"MIR-1K/MIR-1K/Wavfile/{file_pair[0]}")
    audio2, sample_rate = torchaudio.load(f"MIR-1K/MIR-1K/Wavfile/{file_pair[1]}")

    resample_8k = torchaudio.transforms.Resample(sample_rate, target_sr)
    vocals1 = resample_8k(audio1[1])
    vocals2 = resample_8k(audio2[1])
    return vocals1, vocals2
    

def load_and_mix_vocals(file_pairs, target_sr=8000, num_voices=2, sample_len=1):
    """
    Loads audio files from a list of pairs (file_pairs)
    Combines vocals from each pair 
    Reshape mixed vocals and separated vocals into expected input format by Conv-TasNet
    """
    # paired_folder = "paired_vocals/test"
    num_pairs = len(file_pairs)
    num_samples = sample_len*target_sr
    mixed_lst = []
    separated_lst = []
    
    for i, pair in tqdm(enumerate(file_pairs)):
        # print(f"Processing pair {pair}")
        
        # file1_name = pair[0].split('.')[0]
        # file2_name = pair[1].split('.')[0]
        # mixed_name = f"{file1_name}-{file2_name}.wav"
        
        # audio1, sample_rate = torchaudio.load(f"MIR-1K/MIR-1K/Wavfile/{pair[0]}")
        # audio2, sample_rate = torchaudio.load(f"MIR-1K/MIR-1K/Wavfile/{pair[1]}")
    
        # resample_8k = torchaudio.transforms.Resample(sample_rate, target_sr)
        # vocals1 = resample_8k(audio1[1])
        # vocals2 = resample_8k(audio2[1])

        vocals1, vocals2 = load_vocals(pair, target_sr=target_sr)
    
        mixed, separated = combine_vocals(vocals1, vocals2, target_sr, 
                               # save_as=f"{paired_folder}/{mixed_name}"
                              )
        
        dim2 = mixed.shape[1]//num_samples
        # print(mixed.shape, separated.shape)
        mixed = torch.reshape(input=mixed[:, :dim2*num_samples], shape=(dim2, 1, num_samples))
        
        sep1 = separated[:, :dim2*num_samples][0].reshape(dim2, 1, num_samples)
        sep2 = separated[:, :dim2*num_samples][1].reshape(dim2, 1, num_samples)
        separated = torch.cat([sep1, sep2], dim=1)
        # separated = torch.reshape(input=separated[:, :dim2*num_samples], shape=(dim2, num_voices, num_samples))
        # print(mixed.shape, separated.shape)
        mixed_lst.append(mixed)
        separated_lst.append(separated)
        
    return torch.cat(mixed_lst), torch.cat(separated_lst)

def get_metrics(pred, target):
    """
    Calculate the Permutation-invariant SI-SNR and SDR for a given predicted split VS a target split.
    """
    sisnr_pit = PermutationInvariantTraining(scale_invariant_signal_noise_ratio,
                                   mode="speaker-wise", eval_func="max")
    sisnr_pit.cuda()
    sisnr = sisnr_pit(pred, target)

    sdr_pit = PermutationInvariantTraining(signal_distortion_ratio,
                                   mode="speaker-wise", eval_func="max")
    sdr_pit.cuda()
    sdr = sdr_pit(pred, target)
    
    return sisnr.item(), sdr.item()


def evaluate(model, X_test, y_test):
    """
    Evaluation wrapper function
    """

    pred = model(X_test)
    # print(pred.shape)
    # target = y_test.reshape(1, 2, -1) 
    sisnr, sdr = get_metrics(pred, y_test)
    sisnr_orig, sdr_orig = get_metrics(X_test, y_test)
    sisnri = sisnr - sisnr_orig
    sdri = sdr - sdr_orig
    return sisnri, sdri


if __name__ == "__main__":
    # use GPU if available, otherwise, use cpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    files = os.listdir('MIR-1K/MIR-1K/Wavfile')
    num_files = len(files)
    
    test_files = [file for file in files for artist in test_artists if artist in file]
    test_pairs = [pair for pair in itertools.combinations(test_files,2) if pair[0].split('_')[0] != pair[1].split('_')[0]]
    print(f"There are {len(test_pairs)} audio pairs in the test set.")
    
    # random.seed(47)
    sample_len=3
    mixed, sep = load_and_mix_vocals(test_pairs, target_sr=8000, num_voices=2, sample_len=sample_len)
    # mixed, sep = load_and_mix_vocals(random.sample(test_pairs, 100), target_sr=8000, num_voices=2, sample_len=sample_len)

    print(f"There are {len(mixed)} mixed samples of length {sample_len}s in the test set.")
    
    batch_size = 2
    test_dl = DataLoader(TensorDataset(mixed, sep), batch_size=batch_size, shuffle=True)
    print(f"Loaded DataLoader with batch size {batch_size}.")
    
    model = CONVTASNET_BASE_LIBRI2MIX.get_model()
    model = model.to(device)
    print(f"Initialized CONVTASNET_BASE_LIBRI2MIX model.")
    sisnris = []
    sdris = []
    with torch.no_grad():
        for batch_inputs, batch_labels in tqdm(test_dl):
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            # print(batch_inputs.shape)
            # print(batch_labels.shape)
    # random.seed(47)
    # print(random.sample(test_pairs, 12))
    # with torch.no_grad():
            # for pair in random.sample(test_pairs, 12):
                # model = CONVTASNET_BASE_LIBRI2MIX.get_model()
                # model = model.to(device)
                # print(f"Initialized CONVTASNET_BASE_LIBRI2MIX model.")
                # mixed, sep = load_and_mix_vocals([pair], target_sr=8000, num_voices=2, sample_len=sample_len)
                # batch_inputs, batch_labels = mixed.to(device), sep.to(device)
            sisnri, sdri = evaluate(model, batch_inputs, batch_labels)
            # sisnr = sisnr.detach().cpu()
            # sdr = sdr.detach().cpu()
            # print(sisnr, sdr)
            torch.cuda.empty_cache()
            sisnris.append(sisnri)
            sdris.append(sdri)

    print(f"Average SI-SNRi: {np.mean(sisnris)}dB")
    print(f"Average SDRi: {np.mean(sdris)}dB")