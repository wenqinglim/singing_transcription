# Two-vocal Separation and Singing Pitch Transcription

## Set Up:
1. Install dependencies using Poetry: `poetry install`
2. Download the dataset [MIR-1K](https://zenodo.org/records/3532216)

## Running
### Source Separation Evaluation
`poetry run src/source_sep.py`

### Pitch Transcription
Trained models on Hugging Face:
- CREPE-16: [![CREPE-16 Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm-dark.svg)](https://huggingface.co/omgitsqing/CREPE_MIR-1K_16)
- CREPE-24: [![CREPE-24 Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm-dark.svg)](https://huggingface.co/omgitsqing/CREPE_MIR-1K_24)
- CREPE-32: [![CREPE-32 Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm-dark.svg)](https://huggingface.co/omgitsqing/CREPE_MIR-1K_32)

To run training pipeline:
`poetry run src/train_pitch_transcription.py`




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
- `gbqq_lwq_mixed.wav`: mixed track by me (lol) to test in-sync vocals 

Others:
- `analyze_pitch_labels`: initial analysis on distribution of pitches in dataset

## TO-DOs
- [x] Setup Poetry
- [x] Move from notebook to script
- [ ] Test individual scripts in `src/`
- [ ] Get intermediate audio samples for writeup post
- [ ] Create e2e pipeline
- [ ] Host model(s) on HF
- [ ] Add gradio on HF Spaces hehe


