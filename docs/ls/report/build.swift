// macOS-only report renderer; does not import or modify training code.
// Run from the project root: swift -module-cache-path /private/tmp/microduck-swift-cache docs/report/build.swift
import AppKit
import CoreGraphics
import CoreText
import Foundation

struct Block: Decodable {
    let heading: String
    let text: String
    let table: [[String]]?
    let flowDiagram: Bool?
}
struct Page: Decodable { let title: String; let subtitle: String; let blocks: [Block] }
let arguments = Array(CommandLine.arguments.dropFirst())
let submission = arguments.contains("--submission")
var submissionID: String? = nil
if let position = arguments.firstIndex(of: "--id") {
    precondition(submission && position + 1 < arguments.count, "Use --submission --id ACCOUNT_ID")
    let value = arguments[position + 1]
    precondition(value.range(of: "^[A-Za-z0-9_-]+$", options: .regularExpression) != nil, "Use the alphanumeric OmniCampus account ID")
    submissionID = value
}
let source = submission ? "docs/report/content_submission.json" : "docs/report/content.json"
let pages = try JSONDecoder().decode([Page].self, from: Data(contentsOf: URL(fileURLWithPath: source)))
let filename = submissionID.map { "説明資料_PAI最終課題_\($0).pdf" } ?? (submission ? "microduck_report_submission.pdf" : "microduck_report_draft.pdf")
let destination = "output/pdf/" + filename
try FileManager.default.createDirectory(atPath: "output/pdf", withIntermediateDirectories: true)
var media = CGRect(x: 0, y: 0, width: 595.28, height: 841.89)
let documentTitle = "Microduckの強化学習と移動・停止の検証" + (submission ? "" : "（下書き）")
let info: CFDictionary = [kCGPDFContextTitle as String: documentTitle, kCGPDFContextAuthor as String: "", kCGPDFContextCreator as String: "Microduck coursework report"] as CFDictionary
let temporaryDestination = destination + ".building"
let ctx = CGContext(URL(fileURLWithPath: temporaryDestination) as CFURL, mediaBox: &media, info)!
let ink = NSColor(calibratedRed: 0.12, green: 0.19, blue: 0.25, alpha: 1)
let teal = NSColor(calibratedRed: 0.0, green: 0.37, blue: 0.40, alpha: 1)
func draw(_ text: String, x: CGFloat = 44, top: CGFloat, width: CGFloat = 507, size: CGFloat = 10.6, bold: Bool = false, color: NSColor = ink) -> CGFloat {
    let style = NSMutableParagraphStyle()
    style.lineSpacing = 3
    style.paragraphSpacing = 4
    let font = NSFont.systemFont(ofSize: size, weight: bold ? .semibold : .regular)
    let value = NSAttributedString(string: text, attributes: [.font: font, .foregroundColor: color, .paragraphStyle: style])
    let setter = CTFramesetterCreateWithAttributedString(value)
    let bounds = CTFramesetterSuggestFrameSizeWithConstraints(setter, CFRange(location: 0, length: value.length), nil, CGSize(width: width, height: 10000), nil)
    let height = ceil(bounds.height) + 5
    let path = CGPath(rect: CGRect(x: x, y: 841.89 - top - height, width: width, height: height), transform: nil)
    let frame = CTFramesetterCreateFrame(setter, CFRange(location: 0, length: value.length), path, nil)
    precondition(CTFrameGetVisibleStringRange(frame).length == value.length, "Text clipped")
    ctx.saveGState()
    ctx.textMatrix = .identity
    CTFrameDraw(frame, ctx)
    ctx.restoreGState()
    return height
}
func drawTable(_ rows: [[String]], top: CGFloat) -> CGFloat {
    let rowHeight: CGFloat = 23
    let positions: [CGFloat] = [52, 305, 423]
    let widths: [CGFloat] = [246, 110, 120]
    for (rowIndex, row) in rows.enumerated() {
        precondition(row.count == 3, "Expected three table columns")
        let y = top + CGFloat(rowIndex) * rowHeight
        ctx.setFillColor((rowIndex == 0 ? NSColor(calibratedWhite: 0.92, alpha: 1) : NSColor.white).cgColor)
        ctx.fill(CGRect(x: 44, y: 841.89 - y - rowHeight, width: 507, height: rowHeight))
        for (column, cell) in row.enumerated() {
            _ = draw(cell, x: positions[column], top: y + 3, width: widths[column], size: 10.3, bold: rowIndex == 0)
        }
        ctx.setStrokeColor(NSColor(calibratedWhite: 0.85, alpha: 1).cgColor)
        ctx.setLineWidth(0.4)
        ctx.move(to: CGPoint(x: 44, y: 841.89 - y - rowHeight))
        ctx.addLine(to: CGPoint(x: 551, y: 841.89 - y - rowHeight))
        ctx.strokePath()
    }
    return CGFloat(rows.count) * rowHeight + 12
}
func drawFlow(top: CGFloat) -> CGFloat {
    let labels = ["入力\n目標＋真の状態", "通常の制御\n速度指令・選択", "学習済みAI\n関節指令", "MuJoCo\n物理計算", "次の観測\n制御へ戻る"]
    for (index, label) in labels.enumerated() {
        let x = 44 + CGFloat(index) * 103
        let isAI = index == 2
        ctx.setFillColor((isAI ? teal : NSColor(calibratedWhite: 0.94, alpha: 1)).cgColor)
        ctx.fill(CGRect(x: x, y: 841.89 - top - 46, width: 95, height: 46))
        _ = draw(label, x: x + 6, top: top + 7, width: 83, size: 9, bold: true, color: isAI ? .white : ink)
        if index < labels.count - 1 {
            _ = draw("→", x: x + 95, top: top + 15, width: 9, size: 8, color: teal)
        }
    }
    return 58
}
for (index, page) in pages.enumerated() {
    ctx.beginPDFPage(nil)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(cgContext: ctx, flipped: false)
    ctx.setFillColor(teal.cgColor)
    ctx.fill(CGRect(x: 44, y: 797, width: 507, height: 4))
    _ = draw("PHYSICAL AI  /  MICRODUCK", top: 54, size: 9, bold: true, color: teal)
    _ = draw(submission ? "提出用説明資料  |  2026-11-01" : "下書き・結果未確定  |  2026-10-31", x: 335, top: 54, width: 216, size: 9, color: teal)
    var y: CGFloat = 86
    y += draw(page.title, top: y, size: 22, bold: true) + 8
    y += draw(page.subtitle, top: y, size: 10, color: .darkGray) + 19
    for block in page.blocks {
        y += draw(block.heading, top: y, size: 12.5, bold: true, color: teal) + 5
        if let table = block.table { y += drawTable(table, top: y) }
        y += draw(block.text, top: y) + (submission ? 16 : 12)
        if block.flowDiagram == true { y += drawFlow(top: y) }
    }
    if y > 778 { fatalError("Page \(index+1) overflows: \(y)") }
    ctx.setStrokeColor(NSColor.lightGray.cgColor)
    ctx.move(to: CGPoint(x: 44, y: 54)); ctx.addLine(to: CGPoint(x: 551, y: 54)); ctx.strokePath()
    _ = draw(submission ? "学習と評価はシミュレーション内で実施。" : "実測・推定・未実施を区別する。提出前に未確定欄を更新。", top: 797, size: 8, color: .darkGray)
    _ = draw("\(index + 1) / \(pages.count)", x: 515, top: 797, width: 40, size: 8)
    NSGraphicsContext.restoreGraphicsState()
    ctx.endPDFPage()
    print("Page \(index + 1): content bottom \(Int(y)) pt")
}
ctx.closePDF()
if FileManager.default.fileExists(atPath: destination) {
    _ = try FileManager.default.replaceItemAt(URL(fileURLWithPath: destination), withItemAt: URL(fileURLWithPath: temporaryDestination))
} else {
    try FileManager.default.moveItem(atPath: temporaryDestination, toPath: destination)
}
print(destination)
