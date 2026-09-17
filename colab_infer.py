"""Run DeepSeek-OCR in a fresh Python process from the Colab notebook."""

import argparse
from contextlib import ExitStack
from pathlib import Path


PROMPT = "<image>\n<|grounding|>Convert the document to markdown."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepSeek-OCR with vLLM")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="PNG or JPEG image")
    source.add_argument("--pdf", type=Path, help="PDF document (all pages)")
    source.add_argument("--smoke", action="store_true", help="Create a small test image")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def recognize(llm, sampling_params, image) -> str:
    return llm.generate(
        [{"prompt": PROMPT, "multi_modal_data": {"image": image}}],
        sampling_params,
    )[0].outputs[0].text


def main() -> None:
    args = parse_args()

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

        args.output.parent.mkdir(parents=True, exist_ok=True)
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
                    result = recognize(llm, sampling_params, page_image)
                    del page_image
                    output_file.write(f"## Page {page_number}\n\n{result.strip()}\n\n")
                    output_file.flush()
            print(f"Saved {document.page_count} pages: {args.output}")
        else:
            result = recognize(llm, sampling_params, image)
            if not result.strip():
                raise RuntimeError("OCR returned empty text")
            args.output.write_text(result, encoding="utf-8")
            print(result)
            print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
