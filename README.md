_Assesment by EdgeFleet.Ai_
# Cricket Ball Detection
This repository implements a complete computer vision pipeline to detect and track a cricket ball in videos recorded from a single static camera.
## Objective
- Detect the cricket ball centroid in each video frame.
- Track the ball across frames and generate a trajectory.
- Produce reproducible results with clear training, inference, and evaluation steps.

## Features
- Per-frame ball detection with visibility flag.
- CSV annotation output: frame, x, y, visible.
- Processed video with centroid and trajectory overlay.
- Fully reproducible training and inference pipeline.

## Setup
Run the following command to install the required dependencies, which can be found in [requirements.txt](./requirements.txt)
```bash
pip install -r requirements.txt
```

All the required code can be found in the `code` folder. `annotaions` constains CSV annotated files 

## Training

Training scripts are located in `code/training`.

```bash
python code/training/train.py <data_yaml>
```
- Training logs are tracked using Weights & Biases.
- Trained model checkpoints are saved under `runs/detect/`.
- The final trained model used for inference is provided as `code/model.pt`.

Data used for training the model can be found [here](https://drive.google.com/drive/folders/1toUTM7vNPWVGjGVcI9hANmSh_oLyHtpL?usp=sharing).

WandB [dashboard](https://wandb.ai/pranav_ag/cricket-ball-detection/workspace?nw=nwuserpranav_ag) showing training logs and experiments.


`./code/model.pt` contains the model `yolo11m` trained and used for inference.

## Inference and Tracking
Run inference and trajectory generation on a video:
```bash
python code/detection.py <model_path> <input_video>
```
Outputs:

- Annotated CSV file in `annotations/`
- Processed video with centroid and trajectory overlay in `results/`

## Dataset
Data used for training the model can be found [here](https://drive.google.com/drive/folders/1toUTM7vNPWVGjGVcI9hANmSh_oLyHtpL?usp=sharing).

Data used for training the model can be found [here](https://drive.google.com/file/d/1hnaGuqGuMXaFKI5fhfy8gatzCH-6iMcJ/view?usp=sharing).