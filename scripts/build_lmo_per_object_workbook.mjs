import fs from "node:fs/promises";
import path from "node:path";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const [, , csvPathArg, outputPathArg] = process.argv;
if (!csvPathArg || !outputPathArg) {
  console.error("Usage: node build_lmo_per_object_workbook.mjs <input.csv> <output.xlsx>");
  process.exit(1);
}

const csvText = await fs.readFile(csvPathArg, "utf8");
const rows = csvText
  .trim()
  .split(/\r?\n/)
  .map((line) => {
    const parts = [];
    let current = "";
    let inQuotes = false;
    for (let i = 0; i < line.length; i += 1) {
      const ch = line[i];
      if (ch === '"') {
        if (inQuotes && line[i + 1] === '"') {
          current += '"';
          i += 1;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (ch === "," && !inQuotes) {
        parts.push(current);
        current = "";
      } else {
        current += ch;
      }
    }
    parts.push(current);
    return parts;
  });

const headers = rows[0];
const dataRows = rows.slice(1).map((row) => [
  row[0],
  Number(row[1]),
  Number(row[2]),
  Number(row[3]),
  Number(row[4]),
  Number(row[5]),
  Number(row[6]),
  Number(row[7]),
  Number(row[8]),
  Number(row[9]),
]);

const workbook = Workbook.create();
const sheet = workbook.worksheets.add("LMO_Per_Object");
sheet.showGridLines = false;
sheet.freezePanes.freezeRows(1);

sheet.getRange("A1:J1").values = [headers];
sheet.getRange(`A2:J${dataRows.length + 1}`).values = dataRows;

sheet.getRange("A1:J1").format = {
  fill: "#FFFFFF",
  font: { bold: true, color: "#222222", size: 12 },
  borders: {
    bottom: { color: "#C9CDD3", style: "thin" },
  },
};

sheet.getRange(`A2:J${dataRows.length + 1}`).format = {
  fill: "#FFFFFF",
  font: { color: "#222222", size: 11 },
  borders: {
    bottom: { color: "#E9ECEF", style: "thin" },
  },
};

sheet.getRange(`A${dataRows.length + 1}:J${dataRows.length + 1}`).format = {
  font: { bold: true, color: "#222222", size: 11 },
  borders: {
    top: { color: "#C9CDD3", style: "thin" },
    bottom: { color: "#C9CDD3", style: "thin" },
  },
};

sheet.getRange(`B2:B${dataRows.length + 1}`).format.numberFormat = "0";
sheet.getRange(`C2:E${dataRows.length + 1}`).format.numberFormat = "0.0%";
sheet.getRange(`F2:G${dataRows.length + 1}`).format.numberFormat = "0.000";
sheet.getRange(`H2:I${dataRows.length + 1}`).format.numberFormat = "0.00";
sheet.getRange(`J2:J${dataRows.length + 1}`).format.numberFormat = "0.000";

sheet.getRange("A:A").format.columnWidthPx = 180;
sheet.getRange("B:B").format.columnWidthPx = 70;
sheet.getRange("C:E").format.columnWidthPx = 105;
sheet.getRange("F:G").format.columnWidthPx = 110;
sheet.getRange("H:I").format.columnWidthPx = 95;
sheet.getRange("J:J").format.columnWidthPx = 100;
sheet.getRange(`A1:J${dataRows.length + 1}`).format.rowHeightPx = 32;

const preview = await workbook.render({
  sheetName: "LMO_Per_Object",
  autoCrop: "all",
  scale: 1,
  format: "png",
});
await fs.mkdir(path.dirname(outputPathArg), { recursive: true });
await fs.writeFile(outputPathArg.replace(/\.xlsx$/i, ".png"), new Uint8Array(await preview.arrayBuffer()));

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPathArg);
console.log(`saved ${outputPathArg}`);
