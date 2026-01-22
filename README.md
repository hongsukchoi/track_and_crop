# Track and Crop

Track an object in a video and create a cropped video that follows it. Uses [SAM2](https://github.com/facebookresearch/sam2) for object tracking.

## Setup

```bash
# Run the setup script (creates conda env, installs dependencies, downloads model)
./setup.sh

# Activate the environment
conda activate track_and_crop
```

## Usage

```bash
python track_and_crop.py --video input.mp4 --bbox 'x1,y1,x2,y2' --output output.mp4
```

The bounding box should specify the object's location in the first frame as `x1,y1,x2,y2` (top-left and bottom-right corners).

### Example

```bash
python track_and_crop.py \
    --video demo.mp4 \
    --bbox '950,277,1070,500' \
    --output output.mp4 \
    --smoothing 0.8
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--video`, `-v` | Input video path | (required) |
| `--bbox`, `-b` | Initial bounding box `x1,y1,x2,y2` | (required) |
| `--output`, `-o` | Output video path | (required) |
| `--smoothing` | Camera smoothing factor (0=none, 0.9=very smooth) | 0.0 |
| `--crop-scale` | Scale factor for crop region | 1.5 |
| `--aspect-ratio` | Force aspect ratio (e.g., `16:9`, `1:1`) | bbox ratio |
| `--output-size` | Output resolution (e.g., `1280x720`) | auto |
| `--model-size` | SAM2 model: `tiny`, `small`, `base_plus`, `large` | large |
| `--fps` | Output video FPS | same as input |
| `--device` | `cuda` or `cpu` | auto-detect |

## Smoothing

The `--smoothing` option applies EMA (Exponential Moving Average) to the camera motion:

- `0.0` - No smoothing, camera follows object exactly (can be shaky)
- `0.5` - Moderate smoothing
- `0.8` - Smooth camera motion (recommended)
- `0.9` - Very smooth, cinematic feel

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- ~2GB disk space for SAM2 large model checkpoint
