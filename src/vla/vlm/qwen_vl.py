from __future__ import annotations # Used to enable postponed evaluation of type annotations, allowing the use of forward references in type hints.


from dataclasses import dataclass # A lib to simplify the creation of calsses, with automatic generation of special methods like __init__ and __repr__.
from pathlib import Path # Modern way to handle filesystem paths.

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info # Image/video preprocessing fuction provided by Qwen2.5-VL.
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
# Autoprocessor: includes both tokenizer and feature extractor for multimodal inputs -> Tensor.
# Qwen2_5_VLForConditionalGeneration: includes Qwen2.5-VL model (ckpt and structure) for conditional generation tasks, such as text generation based on multimodal inputs.


@dataclass
class QwenVLForwardOutput: # Quickly define a class to hold the output of the forward pass of QwenVL model.
	hidden_states: torch.Tensor
	attention_mask: torch.Tensor
	input_ids: torch.Tensor
# Equivalent to: class QwenVLForwardOutput:
#     def __init__(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, input_ids: torch.Tensor):
#         self.hidden_states = hidden_states
#         self.attention_mask = attention_mask
#         self.input_ids = input_ids

class QwenVL:
	"""Load Qwen2.5-VL and expose its multimodal hidden states."""

	def __init__(
		self,
		model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
		dtype: torch.dtype = torch.bfloat16, # Use bfloat16 for better performance on GPUs that support it, while maintaining numerical stability.
	) -> None:
		if not torch.cuda.is_available():
			raise RuntimeError("CUDA is required to run Qwen2.5-VL.")

		self.device = torch.device("cuda") # Put calculations on GPU for better performance.
		self.dtype = dtype
		self.processor = AutoProcessor.from_pretrained(model_name) # Download relevant tokenizer & feature extractor for model used.
		self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
			model_name,
			torch_dtype=dtype,
			# device_map="auto", # Automatically place model layers on available GPUs.
		).to(self.device) # Load Qwen2.5-VL model and move it to GPU.
		self.model.eval() # Lock the model in evaluation mode, disabling dropout and other training-specific layers.

	@staticmethod
	# Private method to handle image input, converting it to RGB if it's a PIL Image, or returning the path as a string if it's a file path.
	def _image_input(image: Image.Image | str | Path) -> Image.Image | str: 
		if isinstance(image, (str, Path)):
			return str(image) # Return the path (string)
		return image.convert("RGB") # Convert image to RGB mode

	def forward(
		self,
		image: Image.Image | str | Path,
		prompt: str,
	) -> QwenVLForwardOutput:
		messages = [
			{
				"role": "user", # Defines the role of the message sender, which is "user" in this case.
				"content": [
					{
						"type": "image",
						"image": self._image_input(image), # Both direct PIL Image (RGB) and path (string) are accepted.
					},
					{
						"type": "text",
						"text": prompt,
					},
				],
			}
		] # Define the input format for Qwen2.5-VL, which includes an image and a text prompt.
		text = self.processor.apply_chat_template(
			messages,
			tokenize=False,
			add_generation_prompt=True,
		) # Generate a text prompt aligned to training data format, without tokenization, and add a generation prompt for the model to generate a response.
		image_inputs, video_inputs = process_vision_info(messages) # Preprocess the image and video inputs.
		inputs = self.processor(
			text=[text],
			images=image_inputs,
			videos=video_inputs,
			padding=True,
			return_tensors="pt",
		).to(self.device) # Tokenize the text and convert the image/video inputs to tensors, padding them to the same length, and move them to GPU.

		with torch.inference_mode(): # Create a temporary context where gradients are not computed, which saves memory and speeds up inference.
			outputs = self.model(
				**inputs, # Unpack the inputs which are dictionary containing input_ids, attention_mask and pixel_values.
				output_hidden_states=True, # Explictly return all the hidden states, representation vectores of each layer in the model.
			)

		return QwenVLForwardOutput(
			hidden_states=outputs.hidden_states[-1], # Return the hidden states of the last layer, which is typically used for downstream tasks.
			attention_mask=inputs.attention_mask,
			input_ids=inputs.input_ids,
		)


__all__ = ["QwenVL", "QwenVLForwardOutput"] # Only expose QwenVL and QwenVLForwardOutput when using 'from module import *', hiding other internal classes and functions.
