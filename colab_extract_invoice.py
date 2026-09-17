"""Extract a structured invoice from DeepSeek-OCR text in Colab."""

import argparse
import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DIGIT_MAP = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789"
)


class Party(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(description="Name printed for this party, or null")
    vat_number: str | None = Field(description="VAT or tax registration number, or null")
    address: str | None = Field(description="Printed address, or null")


class InvoiceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str | None = Field(description="Product or service description")
    item_code: str | None = Field(description="Printed SKU or item identifier, or null")
    quantity: float | None = Field(description="Quantity, or null if unclear")
    unit_price: float | None = Field(description="Unit price, or null if unclear")
    tax_rate_percent: float | None = Field(description="Tax percentage, or null")
    tax_amount: float | None = Field(description="Tax charged for this item, or null")
    line_total: float | None = Field(description="Total for this item, or null")


class Totals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subtotal: float | None = Field(description="Subtotal before discount and tax, or null")
    discount: float | None = Field(description="Total discount, or null")
    taxable_amount: float | None = Field(description="Amount before tax, or null")
    tax_amount: float | None = Field(description="Total tax amount, or null")
    grand_total: float | None = Field(description="Final invoice total, or null")


class InvoiceData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invoice_number: str | None = Field(description="Invoice serial or number, or null")
    invoice_date: str | None = Field(description="Date exactly as printed, or null")
    supply_date: str | None = Field(description="Supply date exactly as printed, or null")
    reference_number: str | None = Field(description="Reference number, or null")
    seller: Party = Field(description="Seller or issuer")
    buyer: Party = Field(description="Customer or buyer")
    payment_method: str | None = Field(description="Printed payment method, or null")
    currency: str | None = Field(description="Printed currency code, such as SAR, or null")
    items: list[InvoiceItem] = Field(description="Invoice line items; empty if none found")
    totals: Totals = Field(description="Clearly labeled invoice totals")


SYSTEM_PROMPT = """Extract invoice data from OCR text in Arabic and English.
Return one JSON object that matches the required schema.
Use only facts clearly present in the OCR text. Never guess or calculate missing values.
A printed field label without its value means null. Unknown numbers mean null.
Keep names, identifiers, dates, and item descriptions as printed. Do not translate Arabic.
Use JSON numbers for amounts and quantities only when their meaning is clear.
Do not treat a repeated currency code as an amount or a line item.
Use an empty items array if no complete line item can be identified.
Do not include raw OCR text, coordinates, comments, or extra keys."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create structured invoice JSON")
    parser.add_argument("--ocr-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), required=True)
    return parser.parse_args()


def load_ocr(path: Path) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pages = data.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("OCR JSON has no pages")
    if not any(str(page.get("text", "")).strip() for page in pages):
        raise ValueError("OCR returned no readable text")
    return data, pages


def merge_invoice(first: InvoiceData, next_page: InvoiceData) -> None:
    for field_name in (
        "invoice_number",
        "invoice_date",
        "supply_date",
        "reference_number",
        "payment_method",
        "currency",
    ):
        if getattr(first, field_name) is None:
            setattr(first, field_name, getattr(next_page, field_name))

    for party_name in ("seller", "buyer"):
        current = getattr(first, party_name)
        incoming = getattr(next_page, party_name)
        for field_name in ("name", "vat_number", "address"):
            if getattr(current, field_name) is None:
                setattr(current, field_name, getattr(incoming, field_name))

    first.items.extend(next_page.items)
    for field_name in (
        "subtotal",
        "discount",
        "taxable_amount",
        "tax_amount",
        "grand_total",
    ):
        value = getattr(next_page.totals, field_name)
        if value is not None:
            setattr(first.totals, field_name, value)


def review_reasons(invoice: InvoiceData, pages: list[dict]) -> list[str]:
    reasons = []
    if any(page.get("truncated") for page in pages):
        reasons.append("At least one OCR page reached its output limit")
    if invoice.invoice_number is None:
        reasons.append("Invoice number was not identified")
    if invoice.invoice_date is None:
        reasons.append("Invoice date was not identified")
    if invoice.totals.grand_total is None:
        reasons.append("Grand total was not identified")
    if not invoice.items:
        reasons.append("No line items were identified")
    totals = invoice.totals
    if (
        totals.taxable_amount is not None
        and totals.tax_amount is not None
        and totals.grand_total is not None
        and abs(totals.taxable_amount + totals.tax_amount - totals.grand_total) > 0.05
    ):
        reasons.append("Taxable amount plus tax does not match the grand total")
    return reasons


def compact(value: str) -> str:
    value = value.translate(DIGIT_MAP).casefold()
    return re.sub(r"[\W_]+", "", value)


def number_appears(value: float, ocr_text: str) -> bool:
    text = ocr_text.translate(DIGIT_MAP).replace("٬", "").replace("٫", ".")
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    text = re.sub(r"(?<=\d),(?=\d{1,2}(?:\D|$))", ".", text)
    candidates = {str(value), f"{value:.2f}"}
    if value.is_integer():
        candidates.add(str(int(value)))
    return any(
        re.search(r"(?<!\d)" + re.escape(candidate) + r"(?!\d)", text)
        for candidate in candidates
    )


def clear_unsupported_values(invoice: InvoiceData, ocr_text: str) -> list[str]:
    unsupported = []
    searchable = compact(ocr_text)

    def check_text(container, field_name: str, label: str) -> None:
        value = getattr(container, field_name)
        if value is None:
            return
        needle = compact(value)
        if not needle or needle not in searchable:
            setattr(container, field_name, None)
            unsupported.append(label)

    def check_number(container, field_name: str, label: str) -> None:
        value = getattr(container, field_name)
        if value is not None and not number_appears(value, ocr_text):
            setattr(container, field_name, None)
            unsupported.append(label)

    for field_name in (
        "invoice_number",
        "invoice_date",
        "supply_date",
        "reference_number",
        "payment_method",
        "currency",
    ):
        check_text(invoice, field_name, field_name)
    for party_name in ("seller", "buyer"):
        party = getattr(invoice, party_name)
        for field_name in ("name", "vat_number", "address"):
            check_text(party, field_name, f"{party_name}.{field_name}")
    for field_name in (
        "subtotal",
        "discount",
        "taxable_amount",
        "tax_amount",
        "grand_total",
    ):
        check_number(invoice.totals, field_name, f"totals.{field_name}")
    for index, item in enumerate(invoice.items, start=1):
        check_text(item, "description", f"items[{index}].description")
        check_text(item, "item_code", f"items[{index}].item_code")
        if invoice.currency and item.description and compact(item.description) == compact(invoice.currency):
            item.description = None
            unsupported.append(f"items[{index}].description is only a currency code")
        for field_name in (
            "quantity",
            "unit_price",
            "tax_rate_percent",
            "tax_amount",
            "line_total",
        ):
            check_number(item, field_name, f"items[{index}].{field_name}")
    invoice.items = [
        item for item in invoice.items if item.description is not None or item.item_code is not None
    ]

    if unsupported:
        return ["Unsupported invoice values were cleared: " + ", ".join(unsupported)]
    return []


def main() -> None:
    args = parse_args()
    ocr_data, pages = load_ocr(args.ocr_json)

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    print(f"Loading invoice extraction model {MODEL}...", flush=True)
    llm = LLM(
        model=MODEL,
        dtype=args.dtype,
        max_model_len=8192,
        max_num_seqs=1,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=3072,
        structured_outputs=StructuredOutputsParams(json=InvoiceData.model_json_schema()),
    )

    invoice = None
    for page in pages:
        text = str(page.get("text", "")).strip()
        if not text:
            continue
        page_number = page.get("page", "?")
        print(f"Structuring invoice page {page_number}/{len(pages)}", flush=True)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Invoice OCR, page {page_number}:\n\n{text}"},
        ]
        completion = llm.chat(messages, sampling_params)[0].outputs[0]
        if completion.finish_reason == "length":
            raise RuntimeError(f"Invoice extraction for page {page_number} reached its token limit")
        page_invoice = InvoiceData.model_validate_json(completion.text)
        if invoice is None:
            invoice = page_invoice
        else:
            merge_invoice(invoice, page_invoice)

    if invoice is None:
        raise RuntimeError("No invoice data was extracted")

    ocr_text = "\n".join(str(page.get("text", "")) for page in pages)
    reasons = clear_unsupported_values(invoice, ocr_text)
    reasons.extend(review_reasons(invoice, pages))
    result = {
        "document_type": "invoice",
        "source_file": ocr_data.get("source"),
        "page_count": len(pages),
        **invoice.model_dump(),
        "review_required": bool(reasons),
        "review_reasons": reasons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    formatted = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    args.output.write_text(formatted, encoding="utf-8")
    print(formatted, flush=True)
    print(f"Saved structured invoice: {args.output}", flush=True)


if __name__ == "__main__":
    main()
