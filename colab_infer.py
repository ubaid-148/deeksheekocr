"""Run DeepSeek-OCR in a fresh Python process from the Colab notebook."""

import argparse
from pathlib import Path


PROMPT = "<image>\n<|grounding|>Convert the document to markdown."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepSeek-OCR with vLLM")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="PNG or JPEG image")
    source.add_argument("--smoke", action="store_true", help="Create a small test image")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from PIL import Image, ImageDraw, ImageFont
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

    if args.smoke:
        image = Image.new("RGB", (1000, 300), "white")
        draw = ImageDraw.Draw(image)
        font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        font = (
            ImageFont.truetype(str(font_path), 44)
            if font_path.is_file()
            else ImageFont.load_default()
        )
        draw.text((40, 30), "INVOICE", font=font, fill="black")
        draw.text((40, 115), "Item: Notebook", font=font, fill="black")
        draw.text((40, 200), "Total: $42.50", font=font, fill="black")
    else:
        if not args.image.is_file():
            raise FileNotFoundError(args.image)
        with Image.open(args.image) as opened:
            image = opened.convert("RGB")

    llm = LLM(
        model="deepseek-ai/DeepSeek-OCR",
        dtype=args.dtype,
        max_model_len=4096,
        max_num_seqs=1,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        logits_processors=[NGramPerReqLogitsProcessor],
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        extra_args={
            "ngram_size": 30,
            "window_size": 90,
            "whitelist_token_ids": {128821, 128822},
        },
        skip_special_tokens=False,
    )
    result = llm.generate(
        [{"prompt": PROMPT, "multi_modal_data": {"image": image}}],
        sampling_params,
    )[0].outputs[0].text
    if not result.strip():
        raise RuntimeError("OCR returned empty text")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result, encoding="utf-8")
    print(result)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
