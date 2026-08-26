from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml # PyYAML is a YAML parser and emitter for Python, used to read and write YAML configuration files.
from PIL import Image, ImageDraw # Draw 2D graphics on images.


# Allow the script to be run directly from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vla.vlm.qwen_vl import QwenVL


def load_config(config_path: Path) -> dict: # Read yaml comfiguration and return a dictionary.
	with config_path.open("r", encoding="utf-8") as config_file:
		return yaml.safe_load(config_file) or {}


def resolve_dtype(dtype_name: str) -> torch.dtype:
	dtypes = {
		"float16": torch.float16,
		"float32": torch.float32,
		"bfloat16": torch.bfloat16,
	} # Maintain a mapping of string representations of data types to their corresponding torch.dtype objects.
	try:
		return dtypes[dtype_name.lower()] # Retrieve the values in the dictionary mapping.
	except KeyError as error: # Raise not defined error for unsupported data types.
		supported = ", ".join(dtypes)
		raise ValueError(f"Unsupported dtype '{dtype_name}'. Use: {supported}.") from error


def make_demo_image() -> Image.Image:
	"""Create a simple demo image with a red square on a white background."""
	image = Image.new("RGB", (224, 224), "white")
	draw = ImageDraw.Draw(image)
	draw.rectangle((72, 72, 152, 152), fill="red")
	return image 


def resolve_image(image_path: str | None) -> Image.Image | str:
	if image_path is None: # If no image path is provided, generate a demo image.
		return make_demo_image()

	path = Path(image_path)
	if not path.is_absolute():
		path = PROJECT_ROOT / path # Concatenate the absolute path/.
	if not path.exists():
		raise FileNotFoundError(f"Image does not exist: {path}")
	return Image.open(path).convert("RGB") # Load the image and convert it to RGB.


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run one Qwen2.5-VL forward pass and inspect hidden states."
	)
	parser.add_argument(
		"--config",
		type=Path,
		default=PROJECT_ROOT / "configs" / "vlm.yaml",
		help="Path to the VLM YAML configuration.",
	)
	parser.add_argument("--image", help="Optional image path; overrides the YAML value.")
	parser.add_argument("--prompt", help="Optional prompt; overrides the YAML value.")
	return parser.parse_args() # Returns the parsed arguments as a Namespace object, which can be accessed like a dictionary.


def main() -> None:
	args = parse_args()
	config = load_config(args.config)

	if config.get("device", "cuda") != "cuda":
		raise ValueError("This project is configured for NVIDIA CUDA; set device to 'cuda'.")
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver and PyTorch CUDA install.")

    # Can be provided by command or YAML file.
	image = resolve_image(args.image or config.get("image")) 
	prompt = args.prompt or config.get("prompt")
	if not prompt:
		raise ValueError("A prompt is required in the YAML file or through --prompt.")

	vlm = QwenVL(
		model_name=config.get("model_name", "Qwen/Qwen2.5-VL-3B-Instruct"),
		dtype=resolve_dtype(config.get("dtype", "bfloat16")),
	)
	result = vlm.forward(image=image, prompt=prompt)

	print(f"model device: {vlm.device}")
	print(f"model dtype: {vlm.dtype}")
	print(f"hidden states shape: {tuple(result.hidden_states.shape)}")
	print(f"attention mask shape: {tuple(result.attention_mask.shape)}")
	print(f"input ids shape: {tuple(result.input_ids.shape)}")
	print(f"hidden states device: {result.hidden_states.device}")
	print(f"hidden states finite: {bool(torch.isfinite(result.hidden_states).all())}")


if __name__ == "__main__":
	main()
