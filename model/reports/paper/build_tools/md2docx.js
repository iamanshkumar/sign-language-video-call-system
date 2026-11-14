// Converts reports/paper/draft/manuscript.md to .docx (one-off build helper; not part of the repository).
// Usage: node md2docx.js <manuscript.md> <out.docx>
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType,
  ShadingType, ImageRun, AlignmentType, Footer, PageNumber, LevelFormat, BorderStyle, TableLayoutType,
} = require("docx");

const [, , src, out] = process.argv;
const baseDir = path.dirname(path.resolve(src));
const lines = fs.readFileSync(src, "utf8").split("\n");

const FONT = "Times New Roman";
const BODY = 22;            // half-points (11 pt)
const SMALL = 18;           // 9 pt captions
const TABLE_PT = 16;        // 8 pt tables
const CONTENT_W = 11906 - 2 * 1300;   // A4 width minus margins (DXA)

// ---------- inline markdown: **bold**, *italic*, `code`
function runs(text, base = {}) {
  text = text.replace(/(\d) (%|ms\b|s\b)/g, "$1 $2");   // keep numbers and units on one line
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(...runs(t.slice(2, -2), { ...base, bold: true }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), ...base, font: "Courier New", size: (base.size || BODY) - 2 }));
    else out.push(...runs(t.slice(1, -1), { ...base, italics: true }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

function para(text, opts = {}) {
  return new Paragraph({ children: runs(text, opts.run || {}), spacing: { after: 120, line: 276 },
                         alignment: opts.align || AlignmentType.JUSTIFIED, ...opts.p });
}

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), data: b };
}

function splitRow(l) {
  return l.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
}

function table(rowsMd) {
  const rows = rowsMd.filter((r) => !/^\|\s*-/.test(r.trim())).map(splitRow);
  const n = rows[0].length;
  // each column first gets the width of its longest word (headers never split mid-word); the remaining width is
  // shared in proportion to how much longer the column's text is (capped), so long cells wrap instead
  const PAD = 150;   // cell margins (DXA)
  const col = (i) => rows.map((r) => r[i] || "");
  const longest = Array.from({ length: n }, (_, i) => Math.max(...col(i).flatMap((c) => c.split(/\s+/)).map((w) => w.length)));
  // 8 pt (~80 DXA per character); tables whose words do not fit at 8 pt are set in 7 pt (~68 DXA per character)
  let size = TABLE_PT, CH = 80;
  if (longest.reduce((a, w) => a + w * CH + PAD, 0) > CONTENT_W) { size = TABLE_PT - 2; CH = 68; }
  const minW = longest.map((w) => w * CH + PAD);
  const want = Array.from({ length: n }, (_, i) => Math.max(...col(i).map((c) => Math.min(c.length, 24))) * CH + PAD);
  const sumMin = minW.reduce((a, b) => a + b, 0);
  let widths;
  if (sumMin >= CONTENT_W) widths = minW.map((w) => Math.floor((w / sumMin) * CONTENT_W));
  else {
    const extra = want.map((w, k) => Math.max(w - minW[k], 1));
    const te = extra.reduce((a, b) => a + b, 0);
    widths = minW.map((w, k) => Math.floor(w + (extra[k] / te) * (CONTENT_W - sumMin)));
  }
  widths[n - 1] += CONTENT_W - widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
  const borders = { top: border, bottom: border, left: border, right: border };
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths, layout: TableLayoutType.FIXED,
    rows: rows.map((r, ri) => new TableRow({
      tableHeader: ri === 0,
      children: widths.map((w, ci) => new TableCell({
        width: { size: w, type: WidthType.DXA }, borders,
        margins: { top: 40, bottom: 40, left: 60, right: 60 },
        shading: ri === 0 ? { type: ShadingType.CLEAR, color: "auto", fill: "E7E6E6" } : undefined,
        children: [new Paragraph({ children: runs(r[ci] || "", { size, bold: ri === 0 || undefined }) })],
      })),
    })),
  });
}

const children = [];
let i = 0;
let listInstance = 1;   // each numbered list restarts at 1
const isBlank = (l) => l.trim() === "";
const isCaption = (l) => /^\*\*(Table|Figure) \d+\.\*\*/.test(l.trim());

function readParagraph() {
  const buf = [];
  while (i < lines.length && !isBlank(lines[i]) && !/^(#|\||!\[|- |\d+\. |---)/.test(lines[i])) buf.push(lines[i++].trim());
  return buf.join(" ");
}

function caption(text) {
  return para(text, { run: { size: SMALL }, align: AlignmentType.LEFT, p: { spacing: { before: 120, after: 120 }, keepNext: true } });
}

while (i < lines.length) {
  const l = lines[i];
  if (isBlank(l) || l.trim() === "---") { i++; continue; }
  let m;
  if ((m = l.match(/^(#{1,3}) (.*)/))) {
    const lvl = m[1].length;
    if (lvl === 1) children.push(new Paragraph({ children: runs(m[2], { bold: true, size: 32 }), alignment: AlignmentType.CENTER, spacing: { after: 240 } }));
    else children.push(new Paragraph({ heading: lvl === 2 ? HeadingLevel.HEADING_1 : HeadingLevel.HEADING_2, children: runs(m[2]),
                                       spacing: { before: 240, after: 120 }, keepNext: true }));
    i++; continue;
  }
  if (l.startsWith("|")) {
    const rowsMd = [];
    while (i < lines.length && lines[i].startsWith("|")) rowsMd.push(lines[i++]);
    // table captions go above the table: look ahead for the caption paragraph
    let j = i; while (j < lines.length && isBlank(lines[j])) j++;
    if (j < lines.length && /^\*\*Table \d+\.\*\*/.test(lines[j])) {
      i = j; children.push(caption(readParagraph()));
    }
    children.push(table(rowsMd));
    children.push(new Paragraph({ children: [], spacing: { after: 120 } }));
    continue;
  }
  if ((m = l.match(/^!\[[^\]]*\]\(([^)]+)\)/))) {
    const img = pngSize(path.resolve(baseDir, m[1]));
    let wPx = 600, hPx = Math.round((img.h / img.w) * wPx);
    if (hPx > 330) { wPx = Math.round((330 / hPx) * wPx); hPx = 330; }   // keep square plots from filling a page
    children.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true,
      children: [new ImageRun({ type: "png", data: img.data, transformation: { width: wPx, height: hPx } })] }));
    i++; continue;
  }
  if ((m = l.match(/^(- |\d+\. )(.*)/))) {
    const numbered = /^\d/.test(m[1]);
    const buf = [m[2].trim()]; i++;
    while (i < lines.length && /^\s{2,}\S/.test(lines[i])) buf.push(lines[i++].trim());
    children.push(new Paragraph({ children: runs(buf.join(" ")), spacing: { after: 80, line: 276 },
      numbering: { reference: numbered ? "numbers" : "bullets", level: 0, instance: numbered ? listInstance : undefined } }));
    if (!(i < lines.length && /^\d+\. /.test(lines[i])) && numbered) listInstance++;
    continue;
  }
  if (/^\[\d+\] /.test(l)) {   // reference entry: hanging indent
    children.push(para(readParagraph(), { align: AlignmentType.LEFT, run: { size: 20 }, p: { indent: { left: 500, hanging: 500 } } }));
    continue;
  }
  if (isCaption(l)) { children.push(caption(readParagraph())); continue; }
  children.push(para(readParagraph()));
}

const doc = new Document({
  creator: "", title: "Landmark-Based Recognition of 100 Isolated ASL Words",
  styles: {
    default: { document: { run: { font: FONT, size: BODY } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 28, bold: true }, paragraph: { outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, italics: true }, paragraph: { outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1300, bottom: 1300, left: 1300, right: 1300 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: SMALL })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(out, b); console.log("wrote", out, children.length, "blocks"); });
