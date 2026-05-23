from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable


Row = dict[str, Any]
Column = dict[str, Any]


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def default_stringify(value: Any) -> str:
    if value is None:
        return "-"
    return str(value)


def render_rows(rows: list[Row], columns: list[Column]) -> list[dict[str, str]]:
    rendered: list[dict[str, str]] = []
    for row in rows:
        out_row: dict[str, str] = {}
        for col in columns:
            header = str(col["header"])
            key = str(col.get("key", header))
            value = col["value"](row) if "value" in col else row.get(key)
            formatter: Callable[[Any, Row], str] | None = col.get("format")
            text = formatter(value, row) if formatter else default_stringify(value)
            out_row[header] = text
        rendered.append(out_row)
    return rendered


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_rule(align: str) -> str:
    if align == "right":
        return "---:"
    if align == "center":
        return ":---:"
    return ":---"


def write_markdown(path: Path, rows: list[dict[str, str]], columns: list[Column]) -> None:
    headers = [str(col["header"]) for col in columns]
    rules = [markdown_rule(str(col.get("align", "left"))) for col in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(rules) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(header, "").replace("\n", " ") for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def derive_col_spec(columns: list[Column]) -> str:
    token = {"left": "l", "center": "c", "right": "r"}
    inner = "".join(token.get(str(col.get("align", "left")), "l") for col in columns)
    return "@{}" + inner + "@{}"


def tex_cell(text: str, *, bold: bool = False) -> str:
    escaped = latex_escape(text)
    if bold and escaped not in {"", "-"}:
        return r"\textbf{" + escaped + "}"
    return escaped


def write_latex(
    path: Path,
    rows: list[dict[str, str]],
    columns: list[Column],
    *,
    caption: str,
    label: str,
    longtable: bool = False,
    col_spec: str | None = None,
    bold_row_predicate: Callable[[dict[str, str]], bool] | None = None,
    table_env: str = "table*",
    size: str = "small",
) -> None:
    headers = [str(col["header"]) for col in columns]
    spec = col_spec or derive_col_spec(columns)
    header = " & ".join(latex_escape(h) for h in headers) + r" \\"
    body: list[str] = []
    for row in rows:
        is_bold = bool(bold_row_predicate(row)) if bold_row_predicate else False
        body.append(" & ".join(tex_cell(row.get(h, ""), bold=is_bold) for h in headers) + r" \\")

    if longtable:
        text = "\n".join(
            [
                r"\begin{longtable}{" + spec + "}",
                r"\caption{" + latex_escape(caption) + r"}",
                r"\label{" + latex_escape(label) + r"}\\",
                r"\toprule",
                header,
                r"\midrule",
                r"\endfirsthead",
                r"\toprule",
                header,
                r"\midrule",
                r"\endhead",
                *body,
                r"\bottomrule",
                r"\end{longtable}",
                "",
            ]
        )
    else:
        text = "\n".join(
            [
                rf"\begin{{{table_env}}}[t]",
                r"\centering",
                rf"\{size}",
                r"\setlength{\tabcolsep}{4.5pt}",
                r"\renewcommand{\arraystretch}{1.12}",
                r"\caption{" + latex_escape(caption) + r"}",
                r"\label{" + latex_escape(label) + r"}",
                r"\begin{tabular}{" + spec + "}",
                r"\toprule",
                header,
                r"\midrule",
                *body,
                r"\bottomrule",
                r"\end{tabular}",
                rf"\end{{{table_env}}}",
                "",
            ]
        )
    path.write_text(text, encoding="utf-8")


def export_table_bundle(
    base_path: Path,
    rows: list[Row],
    columns: list[Column],
    *,
    caption: str,
    label: str,
    longtable: bool = False,
    col_spec: str | None = None,
    bold_row_predicate: Callable[[dict[str, str]], bool] | None = None,
    table_env: str = "table*",
    size: str = "small",
) -> list[dict[str, str]]:
    rendered = render_rows(rows, columns)
    headers = [str(col["header"]) for col in columns]
    write_csv(base_path.with_suffix(".csv"), rendered, headers)
    write_markdown(base_path.with_suffix(".md"), rendered, columns)
    write_latex(
        base_path.with_suffix(".tex"),
        rendered,
        columns,
        caption=caption,
        label=label,
        longtable=longtable,
        col_spec=col_spec,
        bold_row_predicate=bold_row_predicate,
        table_env=table_env,
        size=size,
    )
    return rendered
