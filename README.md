# Enhancing Person Re-Identification via Uncertainty Feature Fusion and Auto-weighted Measure Combination
## Pipeline
![](images/reidpipe_.png)
## Prerequisites

Before running the script, make sure you have the following dependencies installed:

- [pytorch>=0.4](https://pytorch.org/)
- torchvision

And other additional dependencies and version requirements:

```bash
pip install -r requirements.txt
```

## Datasets
Download links: ([Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [MSMT17](https://arxiv.org/abs/1711.08565), [DukeMTMC-reID](https://arxiv.org/abs/1609.01775))

Change the path to the dataset root by modifying `DATASETS.ROOT_DIR` in the corresponding config file located in the `configs` directory.
## Model Weights
You can find model weights in following repositories:
- **BoT**: [BoT](https://github.com/michuanhaohao/reid-strong-baseline)
- **CLIP-ReID**: [CLIP-ReID](https://github.com/Syliz517/CLIP-ReID/tree/master)

Change the path to the model weights by modifying the `TEST.WEIGHT` in the corresponding configuration file in the `configs` directory. For example, in `configs/market1501.yml`, update the following line:

```yaml
TEST:
  WEIGHT: "path/to/your/model_weights.pth"
```

Ensure that the correct path for the downloaded model weights is provided.

## Test

To test the ReID framework, run the following command:

```bash
python3 tools/test.py --config_file='configs/market1501.yml' --seed 0 --k 4 --n_triple 1000 --baseline bot
```

#### Arguments:
- `--config_file`: Path to the configuration file.
- `--baseline`: Choose the baseline model: "clip" or "bot". Default is "bot".
- `--k`: Top-k similarity based on uncertainty. Default is 5.
- `--n_triple`: Number of data triples used for training. Default is 1000.
- `--seed`: Set a specific random seed for reproducibility. Default is 0.

Additional arguments for the command line interface:
- `--uffm_only`: Use this flag to only apply the Uncertain Feature Fusion Method (UFFM).
- `--out`: Directory where the output will be saved. Default is "output".

The test results may vary slightly (up to ±0.1) from those reported in the paper, depending on the hardware.
## Acknowledgement
Codebase from [BoT](https://github.com/michuanhaohao/reid-strong-baseline) and [CLIP-ReID](https://github.com/Syliz517/CLIP-ReID/tree/master)
## Citation

If you use this code for your research, please cite

```ref
@misc{che2024enhancingpersonreidentificationuncertainty,
      title={Enhancing Person Re-Identification via Uncertainty Feature Fusion and Auto-weighted Measure Combination}, 
      author={Quang-Huy Che and Le-Chuong Nguyen and Vinh-Tiep Nguyen},
      year={2024},
      eprint={2405.01101},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2405.01101}, 
}
```
  







