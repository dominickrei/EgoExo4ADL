<div align="center">
<h5>

<h2><a href="https://arxiv.org/abs/2501.05711" style="color:#9C276A">
From My View to Yours: Learning Egocentric Cues from Exocentric Video using Privileged Egocentric Supervision</a></h2>

[![arXiv](https://img.shields.io/badge/arXiv-2501.05711-b31b1b?style=flat&logo=arxiv)](https://arxiv.org/abs/2501.05711)
[![HuggingFace Ego-in-Exo-Perception](https://img.shields.io/badge/🤗%20HuggingFace-Ego--in--Exo%20Perception-FFD21F?style=flat)](https://huggingface.co/datasets/dreilly/Ego-in-Exo-Perception)
[![HuggingFace ](https://img.shields.io/badge/🤗%20HuggingFace-Training%20Data-FFD21F?style=flat)](https://huggingface.co/datasets/dreilly/Ego2ExoVLM_data)

</h5>

<img width="1954" height="436" alt="teaser_egoexo" src="https://github.com/user-attachments/assets/55272afa-845f-41ee-aa89-d8f6d2bb8822" />

</div>

## 🔔 What’s new
- [Sep 2026] Code and training data is released!
- [Jun 2026] Our work was accepted to **ECCV 2026**!
- [Sep 2025] Released **Ego-in-Exo Perception** benchmark on [HuggingFace](https://huggingface.co/datasets/dreilly/Ego-in-Exo-Perception)!

## ⚙️ Installation
1. Create a conda environment
```shell
conda create --name=ego2exovlm python=3.10
conda activate ego2exovlm
```

2. Clone Ego2ExoVLM and install the required Python packages (we use `torch 2.4.0 + cuda 12.4` in our experiments)
```shell
git clone https://github.com/dominickrei/EgoExo4ADL.git
cd EgoExo4ADL
pip install -r requirements.txt

pip install flash-attn --no-build-isolation
```

## 🏋️ Training Ego2ExoVLM
### 🎥 Preparing Training Data
We provide the instruction pairs as well as the exocentric videos for training through [HuggingFace](https://huggingface.co/datasets/dreilly/Ego2ExoVLM_data). The instruction pairs contain the responses of the egocentric teacher (VideoLLaMA3-7B), paired with the synchronized exocentric videos from the [EgoExo4D](https://ego-exo4d-data.org/) keystep recognition training set.
```shell
huggingface-cli download dreilly/Ego2ExoVLM_data --repo-type dataset --local-dir /path/to/Ego2ExoVLM_data
cd /path/to/Ego2ExoVLM_data && unzip exocentric_videos.zip
```

<details>
<summary>(Optional) Click to view how to regenerate the instruction pairs</summary>

1. Generate the egocentric pseudo-labels with the teacher. For each egocentric video, one query is sampled from each of the descriptive, temporal, and spatial query categories. The videos can be split across multiple processes (e.g., one per GPU)
```shell
python scripts/data/generate_ego_captions.py --proc_id 0 --num_procs 1 --video_dir /path/to/EgoExo4D/keystep_segments --output_dir ./captions
```

2. Merge the pseudo-labels and pair them with an exocentric view of each video
```shell
python scripts/data/build_training_json.py --captions_dir ./captions --video_dir /path/to/EgoExo4D/keystep_segments --output_json /path/to/egoexo4d-exoviews-with-egocaptions.json
```

</details>

### 🔥 Update Training Script and Launch Training
In `scripts/train/ego2exovlm/train_ego2exovlm.sh`, update the following arguments to match your system settings and paths:
* `INIT_MODEL`: This is the path to weights of the base VLM (VideoLLaMA3). Please use the following command to download and save the weights `python scripts/save_basevlm_for_finetuning.py --save-path /path/to/save/base_vlm`
* `DATA_DIR`: The path to the directory containing the exocentric videos (e.g., `/path/to/Ego2ExoVLM_data/exocentric_videos`)
* `TRAINING_JSON`: The path to the json file containing the instruction pairs (e.g., `/path/to/Ego2ExoVLM_data/egoexo4d-exoviews-with-egocaptions.json`)
* (Optional) `NUM_VISUAL_PROBES`: The number of Ego Adaptive Visual Tokens
* (Optional) `INTERACTION_MODULE_POS`: The layers of the vision encoder where the Ego Adaptive Visual Tokens attend to the visual features. Acceptable values are `all` or a comma-separated list of integers (denoting zero-indexed layer indices of the vision encoder)

(**Single node training**) After updating the training script, initiate the training with the following command:
```shell
bash scripts/train/ego2exovlm/train_ego2exovlm.sh 1 <NUM_GPUS>
```

(**Multi-node training with SLURM**) After updating the training script, update the arguments in `ex_multi_node_slurm_job.sh` and submit the job:
```shell
sbatch ex_multi_node_slurm_job.sh
```

**NOTE:** We train with a global batch size of 128. If you run out of GPU memory, reduce `LOCAL_BATCH_SIZE` and the gradient accumulation steps will be increased automatically.

## ❄️ Evaluating Ego2ExoVLM
### 💾 Preparing Evaluation Data
Download [Ego-in-Exo Perception](https://huggingface.co/datasets/dreilly/Ego-in-Exo-Perception) (extract `videos.zip`) and [ADL-X](https://github.com/ADL-X/LLAVIDAL/tree/main?tab=readme-ov-file#quantitative-evaluation-), and organize them with the following structure:

<details>
<summary>Click to view our evaluation directory structure</summary>

    /path/to/vlm_eval_bench/
    ├── adlx
    │   ├── Charades-AR.json
    │   ├── Charades-Description.json
    │   ├── LEMMA-TC.json
    │   ├── Smarthome-AR.json
    │   ├── TSU-Description.json
    │   ├── TSU-TC.json
    │   └── videos
    │       ├── ADLMCQ-TC-TSU
    │       ├── Charades_v1_480
    │       ├── lemma_cropped
    │       └── SH_cropped224x224_better
    └── egoperceptionmcq
        ├── egoinexoperception.json
        └── videos

</details>

### 🏃 Run the evaluations
After downloading the data, update `DATA_ROOT` in `scripts/eval/eval_video.sh` with the path to your evaluation directory, then run the following command:
```shell
bash scripts/eval/eval_video.sh /path/to/trained/ego2exovlm/model <BENCHMARKS> 1 <NUM_GPUS>
e.g., bash scripts/eval/eval_video.sh work_dirs/ego2exovlm/ego2exovlm_qwen2.5_7b_lora egoperceptionmcq,adlx_mcq,adlx_descriptions 1 8
```

Ego-in-Exo Perception results are reported on both the exocentric videos (the Ego2ExoVLM setting) and the egocentric videos.

**NOTE:** The model loader expects `lora` to be in the name of LoRA-trained models (e.g., `ego2exovlm_qwen2.5_7b_lora`), please keep this in the name of your model directory.

**NOTE:** Evaluations on Ego-in-Exo Perception and ADL-X require Llama 3.1. You will need to install [Ollama](https://ollama.com/download) and download the Llama 3.1 model by running the command `ollama run llama3.1` prior to running the evaluations.
* If you are using an HPC environment and can not install Ollama, you will need to run an Ollama server locally
  * To do this, download the Ollama server that matches your system architecture from their [releases page](https://github.com/ollama/ollama/releases). Then update the path to the Ollama server in `scripts/eval/eval_video.sh`

## 🙏 Acknowledgements
We thank the researchers behind the following codebases and model releases for their great open source work which Ego2ExoVLM builds upon! [VisCoP](https://github.com/dominickrei/VisCoP), [VideoLLaMA3](https://github.com/DAMO-NLP-SG/VideoLLaMA3), [EgoExo4D](https://ego-exo4d-data.org/), [LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [SigLIP](https://arxiv.org/abs/2303.15343), and [Qwen2.5](https://arxiv.org/abs/2412.15115).

Please consider citing our work if it is helpful for your project!
```bibtex
@inproceedings{reilly2026ego2exovlm,
  title     = {From My View to Yours: Learning Egocentric Cues from Exocentric Video using Privileged Egocentric Supervision},
  author    = {Dominick Reilly and Manish Kumar Govind and Le Xue and Srijan Das},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```
