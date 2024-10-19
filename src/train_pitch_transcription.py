import argparse

from matplotlib import pyplot as plt
import random 

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torchaudio
import torchaudio.functional as taF
from torch.utils.data import TensorDataset, DataLoader
import scipy
from scipy.io import wavfile

from crepe_model import CREPEModel  

import os

from shared_utils import get_split_by_artist
from train_test_split_list import train_artists, val_artists, test_artists

def audio_to_frames(vocals):
    # make 1024-sample frames of the audio with hop length of 10 milliseconds
    num_samples = len(vocals)
    num_frames = int((num_samples - 1024) / 160) + 1
    frames = vocals.unfold(step=160, size=1024, dimension=0)
    return frames

def pitch_to_frame(annotations, num_frames, num_pitches=410):
    num_classes = num_pitches
    annotation_matrix = torch.zeros((num_frames, num_classes))
    print(annotation_matrix.shape)
    note_range = list(librosa.midi_to_note([i/10 for i in range(360 , 770)], cents=True))
    print(len(note_range))
    for idx in range(num_frames): # iterate each frame, assign pitch label to each frame
        pitch_idx = round(((512+(idx*160))/16000) / 0.02) - 1
        frame_pitch = annotations[pitch_idx]
        if frame_pitch in note_range:
            annotation_matrix[idx, note_range.index(frame_pitch)] = 1
        else:
            # If note it out of range (e.g. 0 values), assign to NA
            annotation_matrix[idx, len(note_range)] = 1
    return annotation_matrix

def load_pitch_labels(file_path):
    pitch_label = np.loadtxt(file_path, dtype=float)
    
    # Convert semitone to discrete note value
    pitch_midi = librosa.midi_to_note(np.round(pitch_label, 1), cents=True)
    pitch_midi[pitch_midi=='C-1+0'] = 'NA'
    return pitch_midi

def load_data(dataset_dir, label_dir):
    '''
     - Load the data into train, val, test based on artist names
     - Extract audio into 1024-length frames with hop size 10ms
     - Annotation: one element corresponding to one audio file
    '''

    files = os.listdir(dataset_dir)
    num_files = len(files)
    num_classes = 410
    
    data = {'train': [],
           'val': [],
           'test': []}
    labels = {'train': [],
           'val': [],
           'test': []}
    
    
    for i, file_id in enumerate(files[:100]):
        print(f"reading {file_id}")
        file_artist = file_id.split('_')[0]
        split = get_split_by_artist(file_artist, train_artists, val_artists, test_artists)
        waveform, sample_rate = torchaudio.load(f"{dataset_dir}/{file_id}")
        vocals = waveform[1]
        # print(vocals)
    
        # make 1024-sample frames of the audio with hop length of 10 milliseconds
        frames = audio_to_frames(vocals)
        # print(frames.shape)
        data[split].append(frames)
        num_frames=len(frames)
        # print(num_frames)
    
        pitch_midi_labels = load_pitch_labels(f"{label_dir}/{file_id.split('.')[0]}.pv")
        # print(pitch_midi_labels)
        annotation_matrix = pitch_to_frame(pitch_midi_labels, num_frames=num_frames, num_pitches=num_classes)
        # print(annotation_matrix.sum())
        labels[split].append(annotation_matrix)

    return data, labels
    

def evaluate(model, data_loader, threshold=0.7, error_range=5):
    model.eval()
    accuracy_all = 0.
    accuracy = 0.
    accuracy_10 = 0.
    accuracy_50 = 0.
    num_gt_pitches = 0
    with torch.no_grad():
        for batch_inputs, batch_labels in data_loader:
            try:
                batch_inputs = torch.reshape(batch_inputs, (batch_inputs.shape[0],1024,1)).to(device)
            except:
                print(batch_inputs.shape)
            batch_labels = batch_labels.to(device)
            batch_outputs = model(batch_inputs).squeeze(dim=1)
            
            # get output prediction indices
            batch_output_arg = torch.argmax(batch_outputs, dim=1)
            
            # Count number of correct predictions (including non-pitch prediction)
            # get labels at predicted indices
            label_values = batch_labels[range(len(batch_output_arg)), batch_output_arg]
            accuracy_all += torch.count_nonzero(label_values)

            # Count number of correct pitch predictions
            pitch_labels = torch.clone(batch_labels)
            pitch_labels[:, -1] = 0
            num_gt_pitches += pitch_labels.sum()
            label_values = pitch_labels[range(len(batch_output_arg)), batch_output_arg]
            accuracy += torch.count_nonzero(label_values)

            # Add error range to labels
            batch_labels_range = torch.clone(pitch_labels)
            
            for i in range(1, error_range+1):
                # Shift label to +- i indices (each index is a 10 cent error range)
                batch_labels_range += pitch_labels.roll(shifts=i, dims=1) 
                batch_labels_range += pitch_labels.roll(shifts=-i, dims=1)
                if i == 1:
                    # Count number of correct pitch predictions with 10-cent error range
                    label_values = batch_labels_range[range(len(batch_output_arg)), batch_output_arg]
                    accuracy_10 += torch.count_nonzero(label_values)

                elif i == 5:
                    # Count number of correct pitch predictions with 50-cent error range
                    label_values = batch_labels_range[range(len(batch_output_arg)), batch_output_arg]
                    accuracy_50 += torch.count_nonzero(label_values)
                
            
    accuracy_all /= len(data_loader.dataset)
    accuracy /= num_gt_pitches
    accuracy_10 /= num_gt_pitches
    accuracy_50 /= num_gt_pitches
    return accuracy_all, accuracy, accuracy_10, accuracy_50

def train(model, train_loader, valid_loader, criterion, optimizer, num_epochs, saved_model, evaluate_every_n_epochs=1):
    model.train()
    num_batches = len(train_loader)
    best_valid_acc = 0.0 # for keeping track of the best accuracy on the validation data
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_inputs, batch_labels in train_loader:
            try:
                batch_inputs = torch.reshape(batch_inputs, (batch_inputs.shape[0],1024,1)).to(device)
            except:
                print(batch_inputs.shape)
            batch_labels = batch_labels.to(device)

            # forward + backward + optimize
            outputs = model(batch_inputs).squeeze(dim=1)
            loss = criterion(outputs, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accumulate loss
            epoch_loss += loss.item()

        epoch_loss /= num_batches
        # print training loss
        metrics['loss'].append(epoch_loss)
        print(f'[{epoch+1}] loss: {epoch_loss:.6f}')
        
        # evaluate the network on the validation data
        if((epoch+1) % evaluate_every_n_epochs == 0):
            valid_acc, valid_acc_10, valid_acc_50 = evaluate(model, valid_loader)
            metrics['val_accuracy'].append(100*valid_acc)
            metrics['val_accuracy_10'].append(100*valid_acc_10)
            metrics['val_accuracy_50'].append(100*valid_acc_50)
            print(f'Validation accuracy: {100*valid_acc:.2f}%; accuracy (+-10c): {100*valid_acc_10:.2f}%; accuracy (+-50c): {100*valid_acc_50:.2f}%')
            
            # if the best validation performance so far, save the network to file 
            if(valid_acc >= best_valid_acc):
                best_valid_acc = valid_acc
                print('Saving best model')
                torch.save(model.state_dict(), saved_model)


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser(description="Train CREPE model on MIR-1K dataset")
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default="MIR-1K/MIR-1K/Wavfile",
        help='Directory of the dataset to be used, containing .wav files.',
        required=False
    )
    parser.add_argument(
        '--label_dir',
        type=str,
        default="MIR-1K/MIR-1K/PitchLabel",
        help='Directory of vocal pitch labels.',
        required=False,
    )
    parser.add_argument(
        '--split_dir',
        type=str,
        default="audio/split_vocals",
        help='Directory to save the split audio files.',
        required=False,
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default="metrics",
        help='Directory to save the evaluation results.',
        required=False,
    )
    parser.add_argument(
        '--huggingface_model',
        type=str,
        default="omgitsqing/CREPE_MIR-1K_16",
        help='Huggingface model name to save the model to.',
        required=False,
    )
    args = parser.parse_args()

    # use GPU if available, otherwise, use cpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # files = os.listdir('MIR-1K/MIR-1K/Wavfile')
    # files = os.listdir(args.dataset_dir)
    # num_files = len(files)
    # num_classes = 410
    
    # '''
    #  - Load the data into train, val, test based on artist names
    #  - Extract audio into 1024-length frames with hop size 10ms
    #  - Annotation: one element corresponding to one audio file
    # '''
    
    # data = {'train': [],
    #        'val': [],
    #        'test': []}
    # labels = {'train': [],
    #        'val': [],
    #        'test': []}
    
    
    # for i, file_id in enumerate(files[:100]):
    #     print(f"reading {file_id}")
    #     file_artist = file_id.split('_')[0]
    #     split = get_split_by_artist(file_artist, train_artists, val_artists, test_artists)
    #     waveform, sample_rate = torchaudio.load(f"MIR-1K/MIR-1K/Wavfile/{file_id}")
    #     vocals = waveform[1]
    #     # print(vocals)
    
    #     # make 1024-sample frames of the audio with hop length of 10 milliseconds
    #     frames = audio_to_frames(vocals)
    #     # print(frames.shape)
    #     data[split].append(frames)
    #     num_frames=len(frames)
    #     # print(num_frames)
    
    #     pitch_midi_labels = load_pitch_labels(f"MIR-1K/MIR-1K/PitchLabel/{file_id.split('.')[0]}.pv")
    #     # print(pitch_midi_labels)
    #     annotation_matrix = pitch_to_frame(pitch_midi_labels, num_frames=num_frames, num_pitches=num_classes)
    #     # print(annotation_matrix.sum())
    #     labels[split].append(annotation_matrix)

    data, labels = load_data(args.dataset_dir, args.label_dir)

    
    # data = torch.cat(data_lst)
    # labels = torch.cat(labels_lst)
    
    train_data = torch.cat(data['train'])
    valid_data = torch.cat(data['val'])
    test_data = torch.cat(data['test'])
    
    train_labels = torch.cat(labels['train'])
    valid_labels = torch.cat(labels['val'])
    test_labels = torch.cat(labels['test'])
    
    print(train_data.shape, valid_data.shape, test_data.shape)
    print(train_labels.shape, valid_labels.shape, test_labels.shape)
    
    
    # Learn statistics from train set only!
    train_mean = train_data.mean(axis=(0, 1), keepdims=True)
    train_std = train_data.std(axis=(0, 1), keepdims=True)
    # Apply train set statistics to all splits
    train_data = torch.clip((train_data - train_mean) / train_std, min=1e-8, max=None)
    valid_data = torch.clip((valid_data - train_mean) / train_std, min=1e-8, max=None)
    test_data = torch.clip((test_data - train_mean) / train_std, min=1e-8, max=None)
    print(train_data.shape, valid_data.shape, test_data.shape)
    
    batch_size = 20
    
    train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(valid_data, valid_labels), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=batch_size, shuffle=False)
    
    num_epochs = 10 # the number of training epoch (when you've gone through all samples of the training data, that's one epoch)
    saved_model = './best_model.pkl' # path for saving the best model during training
    evaluate_every_n_epochs = 1 # how often you want to evaluate the network during training?
    
    metrics = {"loss":[],
              "val_accuracy": [],
               "val_accuracy_10": [],
               "val_accuracy_50": [],
              
              }
    
    model = CREPEModel().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    train(model, train_loader, valid_loader, criterion, optimizer, num_epochs, saved_model, evaluate_every_n_epochs)

    if args.huggingface_model:
        model.push_to_hub("omgitsqing/CREPE_MIR-1K_16")

    print(metrics)

    # Save all results in json file
    with open(f"{args.results_dir}/pitch_transcription_results.json", "w") as f:
        json.dump(metrics, f)
        print(f"Saved evaluation results to: {args.results_dir}/{args.dataset}/pitch_transcription_results.json")
