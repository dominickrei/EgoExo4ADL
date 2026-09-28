"""
Build the Ego2ExoVLM training json from the teacher captions produced by `scripts/data/generate_ego_captions.py`.

The captions of each egocentric video are merged across processes and paired with one synchronized exocentric view of
the same keystep segment, so the student observes the exocentric video while being supervised by the egocentric teacher
(Ego2Exo Sequence Distillation). Each sample has the following format:
    {
        "id": "<segment id>",
        "video": "<segment id>_exo<N>.mp4",
        "conversations": [
            {"from": "human", "value": "<video>\n<query>"},
            {"from": "gpt", "value": "<egocentric teacher response>"}
        ]
    }

Usage:
    python scripts/data/build_training_json.py \
        --captions_dir ./captions \
        --video_dir /path/to/EgoExo4D/keystep_segments \
        --output_json /path/to/ego2exovlm_training_data/egoexo4d-exoviews-with-egocaptions.json
"""
import argparse, glob, json, os, random

EGO_SUFFIX = "_ego.mp4"

argparser = argparse.ArgumentParser()
argparser.add_argument("--captions_dir", type=str, default="./captions", help="Directory containing captions_proc_*.json")
argparser.add_argument("--video_dir", type=str, default="/path/to/EgoExo4D/keystep_segments", help="Directory containing the EgoExo4D keystep segments")
argparser.add_argument("--output_json", type=str, required=True)
argparser.add_argument("--seed", type=int, default=0)
args = argparser.parse_args()

random.seed(args.seed)

## Merge the captions from all processes
caption_files = sorted(glob.glob(os.path.join(args.captions_dir, "captions_proc_*.json")))
assert caption_files, f"No caption files found in {args.captions_dir}"

captions = {}
for caption_file in caption_files:
    with open(caption_file, 'r') as f:
        captions.update(json.load(f))

print(f"Loaded captions for {len(captions)} egocentric videos from {len(caption_files)} files")

## Index the exocentric views of each segment, e.g., <segment id>_exo1.mp4, <segment id>_exo2.mp4, ...
exo_views = {}
for video_name in sorted(os.listdir(args.video_dir)):
    if not video_name.endswith(".mp4") or "_exo" not in video_name:
        continue
    segment_id = video_name.rsplit("_exo", 1)[0]
    exo_views.setdefault(segment_id, []).append(video_name)

## Pair the egocentric captions with an exocentric view of the same segment
samples = []
num_failed, num_no_exo = 0, 0
for ego_video_name in sorted(captions):
    qa_pairs = captions[ego_video_name]

    # skip videos where caption generation failed
    if not isinstance(qa_pairs, dict) or not qa_pairs:
        num_failed += 1
        continue

    segment_id = ego_video_name[:-len(EGO_SUFFIX)]
    exo_videos = exo_views.get(segment_id, [])
    if not exo_videos:
        num_no_exo += 1
        continue

    exo_video_name = random.choice(exo_videos)

    for question, answer in qa_pairs.items():
        # randomly place the video token before or after the query
        if random.random() < 0.5:
            prompt = f"<video>\n{question}"
        else:
            prompt = f"{question}\n<video>"

        samples.append({
            "id": segment_id,
            "video": exo_video_name,
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": answer},
            ]
        })

print(f"Skipped {num_failed} videos with failed caption generation and {num_no_exo} videos without an exocentric view")
print(f"Created {len(samples)} samples from {len(set(s['id'] for s in samples))} segments")

os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
with open(args.output_json, 'w') as f:
    json.dump(samples, f, indent=2)

print(f"Saved training json to {args.output_json}")
