# Two-vocal Separation and Singing Pitch Transcription

## File Descriptions:
Training scripts:
- `convtasnet.ipynb`: Evaluate pre-trained Conv-TasNet model for vocal sepration
- `test_crepe.ipynb` : Train CREPE model for pitch transcription
- `crepe_model.py`: Define CREPE model

Pipeline:
- `combined_pipeline.ipynb`: combines both Conv-TasNet and trained CREPE models into a single pipeline
- `crepe_model.py`

Artifacts:
- `best_crepe_xx.pkl`: Stores best trained CREPE model for each model size
- `leon_7_jmzen_5.wav`: mixed track used for case study
- `gbqq_lwq_mixed.wav`: mixed track by me to test in-sync vocals 

Others:
- `analyze_pitch_labels`: initial analysis on distribution of pitches in dataset