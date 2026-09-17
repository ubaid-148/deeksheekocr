"""Run DeepSeek-OCR in a fresh Python process from the Colab notebook."""

import argparse
import json
import re
from contextlib import ExitStack
from pathlib import Path


PROMPT = "<image>\nFree OCR."
MODEL = "deepseek-ai/DeepSeek-OCR"
GROUNDING_BLOCK = re.compile(
    r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>", re.DOTALL
)
SPECIAL_TOKEN = re.compile(r"<\|[^<>]*\|>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepSeek-OCR with vLLM")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="PNG or JPEG image")
    source.add_argument("--pdf", type=Path, help="PDF document (all pages)")
    source.add_argument("--smoke", action="store_true", help="Create a small test image")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, help="Clean per-page JSON result")
    return parser.parse_args()


def clean_ocr_text(raw: str) -> str:
    text = GROUNDING_BLOCK.sub("", raw)
    text = SPECIAL_TOKEN.sub("", text)
    text = text.replace("<｜end▁of▁sentence｜>", "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def recognize(llm, sampling_params, image) -> tuple[str, bool]:
    completion = llm.generate(
        [{"prompt": PROMPT, "multi_modal_data": {"image": image}}],
        sampling_params,
    )[0].outputs[0]
    return clean_ocr_text(completion.text), completion.finish_reason == "length"


def main() -> None:
    args = parse_args()
    print("Loading OCR dependencies...", flush=True)

    from PIL import Image, ImageDraw, ImageFont
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

    with ExitStack() as resources:
        document = None
        if args.pdf:
            if not args.pdf.is_file():
                raise FileNotFoundError(args.pdf)
            import pymupdf

            document = resources.enter_context(pymupdf.open(str(args.pdf)))
            if document.needs_pass:
                raise ValueError("Password-protected PDFs are not supported")
            if document.page_count == 0:
                raise ValueError("The PDF has no pages")
            print(f"PDF pages: {document.page_count}", flush=True)
        elif args.smoke:
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

        print("Loading DeepSeek-OCR model; the first run downloads its weights...", flush=True)
        llm = LLM(
            model=MODEL,
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
            max_tokens=2048,
            extra_args={
                "ngram_size": 30,
                "window_size": 90,
                "whitelist_token_ids": {128821, 128822},
            },
            skip_special_tokens=False,
        )
        print("Model ready", flush=True)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        pages = []
        if document is not None:
            with args.output.open("w", encoding="utf-8") as output_file:
                for page_number, page in enumerate(document, start=1):
                    print(f"OCR page {page_number}/{document.page_count}", flush=True)
                    pix = page.get_pixmap(
                        dpi=150, colorspace=pymupdf.csRGB, alpha=False
                    )
                    page_image = Image.frombytes(
                        "RGB", (pix.width, pix.height), pix.samples
                    )
                    del pix
                    result, truncated = recognize(llm, sampling_params, page_image)
                    del page_image
                    pages.append(
                        {"page": page_number, "text": result, "truncated": truncated}
                    )
                    output_file.write(f"## Page {page_number}\n\n{result}\n\n")
                    output_file.flush()
                    if truncated:
                        print(
                            f"Warning: Page {page_number} reached the output token limit",
                            flush=True,
                        )
            print(f"Saved {document.page_count} pages: {args.output}")
        else:
            result, truncated = recognize(llm, sampling_params, image)
            if not result.strip():
                raise RuntimeError("OCR returned empty text")
            pages.append({"page": 1, "text": result, "truncated": truncated})
            args.output.write_text(result, encoding="utf-8")
            if args.smoke:
                print(result)
            if truncated:
                print("Warning: OCR reached the output token limit", flush=True)
            print(f"Saved OCR text: {args.output}", flush=True)

        if args.json_output:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(
                json.dumps(
                    {
                        "source": args.pdf.name if args.pdf else args.image.name if args.image else "smoke",
                        "model": MODEL,
                        "mode": "free_ocr",
                        "pages": pages,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"Saved JSON: {args.json_output}", flush=True)


if __name__ == "__main__":
    main()
