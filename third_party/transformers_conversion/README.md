# Mask2Former conversion reference

`convert_mask2former_original_v4_37_2.py` is an unchanged copy of Hugging Face Transformers v4.37.2, licensed under Apache-2.0 as stated in the source header.

Source: https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/mask2former/convert_mask2former_original_pytorch_checkpoint_to_pytorch.py

Git blob: `ea1c578509f60bb6fcb07a373d82635188444dc8`.

`tools/verify_mask2former_conversion.py` verifies this identity and executes only its mapping class for an independent round-trip check. It does not execute the reference module's imports, download helpers or command line entry point.
