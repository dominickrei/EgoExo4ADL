"""
Generate egocentric pseudo-labels with the frozen teacher (VideoLLaMA3-7B).

For every egocentric EgoExo4D keystep segment, one query is sampled from each of the descriptive, temporal, and spatial
query categories and answered by the teacher using the egocentric video. The answers are later paired with the
synchronized exocentric videos by `scripts/data/build_training_json.py` to supervise the student.

Videos can be split across multiple processes (e.g., one per GPU):
    python scripts/data/generate_ego_captions.py --proc_id 0 --num_procs 4 --video_dir /path/to/EgoExo4D/keystep_segments
"""
import argparse, random, torch, tqdm, json, glob, os
from transformers import AutoModelForCausalLM, AutoProcessor

argparser = argparse.ArgumentParser()
argparser.add_argument("--proc_id", type=int, required=True)
argparser.add_argument("--num_procs", type=int, required=True)
argparser.add_argument("--video_dir", type=str, default="/path/to/EgoExo4D/keystep_segments", help="Directory containing the EgoExo4D keystep segments")
argparser.add_argument("--save_interval", type=int, default=10, help="Save results every N videos")
argparser.add_argument("--output_dir", type=str, default="./captions", help="Directory to save results")
argparser.add_argument("--seed", type=int, default=0)
args = argparser.parse_args()

random.seed(args.seed + args.proc_id)

def generate_caption(video_path, question, model, processor):
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": 180}},
                {"type": "text", "text": question},
            ]
        },
    ]

    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    output_ids = model.generate(**inputs, max_new_tokens=1024)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return response

def load_existing_results(output_file):
    """Load existing results if the file exists"""
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"Warning: Could not load existing results from {output_file}")
    return {}

def save_results(results, output_file):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

## Load teacher model
device = "cuda:0"
model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"
print(f"Loading model on device {device}...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map={"": device},
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
print("Model loaded successfully!")

## Get egocentric video files and distribute them across processes
video_pattern = os.path.join(args.video_dir, "*_ego.mp4")
all_videos = sorted(glob.glob(video_pattern))

assert all_videos, f"No videos found at {video_pattern}"

print(f"Found {len(all_videos)} total videos")

# distribute videos across processes
videos_per_proc = len(all_videos) // args.num_procs
start_idx = args.proc_id * videos_per_proc
if args.proc_id == args.num_procs - 1:  # last process handles any remaining videos
    end_idx = len(all_videos)
else:
    end_idx = start_idx + videos_per_proc

my_videos = all_videos[start_idx:end_idx]
print(f"Process {args.proc_id}: Processing {len(my_videos)} videos (indices {start_idx}-{end_idx-1})")

output_file = os.path.join(args.output_dir, f"captions_proc_{args.proc_id}.json")
results = load_existing_results(output_file)

print(f"Output will be saved to: {output_file}")
print(f"Already processed: {len(results)} videos")

# question categories
descriptive_qs = [
    "Describe what is happening in this video in detail.",
    "What events are taking place in this video in detail?",
    "Describe the main actions shown in the video in detail.",
    "What activity is being performed in the video in detail?",
    "Provide a detailed description of what occurs in this clip."
]

temporal_qs = [
    "Summarize the sequence of events shown in this video.",
    "Outline the order of events shown in this video.",
    "Summarize the progression of events from beginning to end."
]

spatial_qs = [
    "What are the primary objects in the scene relevant to the action being performed?",
    "Where are the main objects and people located in the scene?",
    "Describe the objects in the video and how they are used.",
    "What objects are visible throughout the video?"
]

question_types = [descriptive_qs, temporal_qs, spatial_qs]

# generate captions
processed_count = 0
skipped_count = 0

with tqdm.tqdm(total=len(my_videos)*len(question_types)) as pbar:
    for i, video_path in enumerate(my_videos):
        video_name = os.path.basename(video_path)

        if video_name in results:
            skipped_count += 1
            pbar.update(len(question_types))
            continue

        results[video_name] = {}

        questions = [random.choice(x) for x in question_types]

        for question in questions:
            try:
                caption = generate_caption(video_path, question, model, processor)
                results[video_name][question] = caption
                processed_count += 1

                # auto-save
                if processed_count % args.save_interval == 0:
                    save_results(results, output_file)

                pbar.update(1)

            except Exception as e:
                print(f"Error processing {video_name}: {str(e)}")
                results[video_name] = f"ERROR: {str(e)}"
                processed_count += 1
                pbar.update(1)

        pbar.set_description(f"Video {i+1}/{len(my_videos)}")

# final save
save_results(results, output_file)
print(f"Process {args.proc_id} completed!")
print(f"Total processed: {processed_count}")
print(f"Total skipped (already done): {skipped_count}")
print(f"Results saved to: {output_file}")
