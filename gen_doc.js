const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat,
  HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageNumber, PageBreak, TableOfContents
} = require("docx");

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 60, bottom: 60, left: 100, right: 100 };
const headerBg = { fill: "1B4F72", type: ShadingType.CLEAR };
const lightBg = { fill: "EBF5FB", type: ShadingType.CLEAR };

// Helper: table cell
function tc(text, width, bold, bg) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: bg || undefined,
    margins: cellMargins,
    children: [new Paragraph({
      children: [new TextRun({ text, bold: !!bold, font: "Microsoft YaHei", size: bold ? 22 : 20 })]
    })]
  });
}

// Helper: header cell
function thc(text, width) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: headerBg,
    margins: cellMargins,
    children: [new Paragraph({
      children: [new TextRun({ text, bold: true, font: "Microsoft YaHei", size: 20, color: "FFFFFF" })]
    })]
  });
}

// Helper: normal paragraph
function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120 },
    children: [new TextRun({ text, font: "Microsoft YaHei", size: 22, ...opts })]
  });
}

// Helper: heading
function h(text, level) {
  return new Paragraph({
    heading: level === 1 ? HeadingLevel.HEADING_1 : HeadingLevel.HEADING_2,
    children: [new TextRun({ text, font: "Microsoft YaHei", bold: true,
      size: level === 1 ? 36 : 30, color: "1B4F72" })]
  });
}

// Helper: bullet point
function bullet(text, ref = "bullets") {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    spacing: { after: 60 },
    children: [new TextRun({ text, font: "Microsoft YaHei", size: 22 })]
  });
}

// ── Build Document ──
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Microsoft YaHei", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Microsoft YaHei", color: "1B4F72" },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: "Microsoft YaHei", color: "2874A6" },
        paragraph: { spacing: { before: 240, after: 160 }, outlineLevel: 1 } },
    ]
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u25CF",
          alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",
          alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ]
  },
  sections: [
    // ===== COVER PAGE =====
    {
      properties: {
        page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } }
      },
      children: [
        new Paragraph({ spacing: { before: 3600 } }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
          children: [new TextRun({ text: "地理空间TIFF机器学习平台", font: "Microsoft YaHei", size: 56, bold: true, color: "1B4F72" })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 },
          children: [new TextRun({ text: "Geospatial TIFF ML Platform", font: "Arial", size: 32, color: "5DADE2", italics: true })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
          children: [new TextRun({ text: "使用说明书 v2.2", font: "Microsoft YaHei", size: 28, color: "7FB3D8" })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 400 },
          children: [new TextRun({ text: "栅格数据机器学习建模 | 12种算法 | 年/月数据支持", font: "Microsoft YaHei", size: 22, color: "666666" })] }),
        new Paragraph({ alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "空间分析 | 模型集成 | SHAP可解释性 | 预测输出", font: "Microsoft YaHei", size: 22, color: "666666" })] }),
        new Paragraph({ spacing: { before: 2400 }, alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "2026年5月", font: "Microsoft YaHei", size: 22, color: "999999" })] }),
      ]
    },

    // ===== TABLE OF CONTENTS =====
    {
      properties: {
        page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } }
      },
      children: [
        h("目录", 1),
        new TableOfContents("目录", { hyperlink: true, headingStyleRange: "1-2" }),
        new Paragraph({ children: [new PageBreak()] }),
      ]
    },

    // ===== CHAPTER 1: OVERVIEW =====
    {
      properties: {
        page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } }
      },
      headers: {
        default: new Header({ children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "1B4F72", space: 4 } },
          children: [new TextRun({ text: "地理空间TIFF机器学习平台 - 使用说明书", font: "Microsoft YaHei", size: 16, color: "999999", italics: true })]
        })] })
      },
      footers: {
        default: new Footer({ children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "- ", font: "Microsoft YaHei", size: 16, color: "999999" }),
                     new TextRun({ children: [PageNumber.CURRENT], font: "Microsoft YaHei", size: 16, color: "999999" }),
                     new TextRun({ text: " -", font: "Microsoft YaHei", size: 16, color: "999999" })]
        })] })
      },
      children: [
        h("1. 软件简介", 1),
        p("地理空间TIFF机器学习平台是一款基于Streamlit开发的桌面端Web应用，专门用于栅格数据的机器学习建模与预测。它支持对逐年/逐月TIFF格式的地理空间数据进行多种机器学习模型的训练、评估、预测和可解释性分析。"),
        p("本软件面向地理学、遥感、气象、生态学等领域的研究者和数据分析人员，提供从数据加载到模型输出的一站式分析流程。"),
        new Paragraph({ spacing: { before: 200 } }),
        h("1.1 核心功能", 2),
        bullet("支持12种机器学习模型：OLS、Ridge、Lasso、ElasticNet、RF、XGBoost、LightGBM、ExtraTrees、GBR、SVR、MLP、KNN"),
        bullet("年度数据和月度数据双模式，支持嵌套目录结构"),
        bullet("模拟模式和外推预测双模式（区分有无未来X数据）"),
        bullet("模型持久化保存与复用"),
        bullet("SHAP模型可解释性分析"),
        bullet("逐像元时间序列分析"),
        bullet("自动可视化图表生成（散点图、残差图、RMSE箱线图、特征重要性等）"),
        bullet("HTML/PDF评估报告自动生成"),
        bullet("完整的表格数据导出（逐像素指标、特征重要性CSV）"),
        new Paragraph({ spacing: { before: 200 } }),
        h("1.2 系统要求", 2),
        bullet("操作系统：Windows 10/11"),
        bullet("Python 3.12 及以上"),
        bullet("内存：建议 8GB 及以上"),
        bullet("硬盘：至少 500MB 可用空间"),
        new Paragraph({ children: [new PageBreak()] }),

        h("2. 快速入门", 1),
        h("2.1 启动软件", 2),
        p("双击软件目录下的「启动应用.vbs」即可无黑窗启动。首次启动会自动打开浏览器访问 http://localhost:8501。"),
        p("也可使用「启动应用.bat」命令行方式启动。"),
        new Paragraph({ spacing: { before: 200 } }),
        h("2.2 基本工作流程", 2),
        p("典型的分析流程如下："),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 60 },
          children: [new TextRun({ text: "输入数据路径：在左侧边栏输入Y（因变量）TIFF文件夹和X（自变量）TIFF文件夹", font: "Microsoft YaHei", size: 22 })] }),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 60 },
          children: [new TextRun({ text: "选择数据模式：年度数据或月度数据", font: "Microsoft YaHei", size: 22 })] }),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 60 },
          children: [new TextRun({ text: "确认时间范围：软件自动检测年份范围，可手动调整训练/预测起止年", font: "Microsoft YaHei", size: 22 })] }),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 60 },
          children: [new TextRun({ text: "选择模型：勾选需要训练的机器学习模型（默认全不选）", font: "Microsoft YaHei", size: 22 })] }),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 60 },
          children: [new TextRun({ text: "点击「开始训练」→ 等待训练完成", font: "Microsoft YaHei", size: 22 })] }),
        new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 60 },
          children: [new TextRun({ text: "查看结果：训练完成后自动切换到结果分析页面", font: "Microsoft YaHei", size: 22 })] }),
        new Paragraph({ children: [new PageBreak()] }),

        h("3. 功能页面详解", 1),
        h("3.1 首页与运行", 2),
        p("这是核心工作页面，完成所有训练配置和启动。"),
        new Paragraph({ spacing: { before: 120 } }),
        p("数据模式选择：", { bold: true }),
        bullet("年度数据：读取形如 2000.tif 的文件"),
        bullet("月度数据：支持两种目录结构 —— 平铺式（2000-01.tif）和嵌套式（2000/01.tif）"),
        new Paragraph({ spacing: { before: 120 } }),
        p("预测模式选择：", { bold: true }),
        bullet("模拟模式：预测年份的X自变量数据已存在，使用实际X值预测"),
        bullet("外推预测：预测年份无X数据，使用训练末年X值替代"),
        new Paragraph({ spacing: { before: 120 } }),
        p("模型选择：", { bold: true }),
        p("所有模型默认不勾选，需手动选择。每个模型可展开调整超参数。12种模型分为三类："),
        bullet("线性模型：OLS、Ridge、Lasso、ElasticNet —— 简单快速、可解释性强"),
        bullet("树模型：RF、XGBoost、LightGBM、ExtraTrees、GBR —— 非线性建模、特征重要性"),
        bullet("其他模型：SVR、MLP、KNN —— 支持向量回归、神经网络、K近邻"),

        new Paragraph({ children: [new PageBreak()] }),
        h("3.2 数据探索", 2),
        p("在正式训练前进行数据质量检查。功能包括："),
        bullet("文件信息列表：显示前5个TIFF文件的基本信息（分辨率、波段数、坐标系等）"),
        bullet("栅格预览图：将TIFF渲染为彩色图像快速查看"),
        bullet("数值分布直方图：查看数据分布形态"),
        bullet("时间序列趋势图：随机采样像素展示Y值随时间变化"),
        bullet("时间自相关热力图：展示各年份之间的相关性"),

        new Paragraph({ spacing: { before: 200 } }),
        h("3.3 结果分析", 2),
        p("训练完成后显示6个标签页："),
        bullet("性能总览：模型评估指标对比表（R²、RMSE、MAE），高亮最佳模型"),
        bullet("可视化图表：散点图、残差直方图、RMSE箱线图、特征重要性柱状图等"),
        bullet("空间分析：逐像素性能空间分布图"),
        bullet("详细指标：每个模型的逐像素指标详情和分布直方图"),
        bullet("评估报告：HTML报告预览和下载，PDF报告生成"),
        bullet("导出结果：CSV指标表下载、JSON/YAML配置导出"),

        new Paragraph({ spacing: { before: 200 } }),
        h("3.4 模型复用", 2),
        p("训练完成后模型自动保存到 outputs/models/ 目录（joblib格式）。在模型复用页面："),
        bullet("自动列出所有已保存的模型包"),
        bullet("加载任意模型包后用新X数据进行预测"),
        bullet("无需重新训练，快速产出新预测TIFF"),

        new Paragraph({ spacing: { before: 200 } }),
        h("3.5 SHAP解释", 2),
        p("独立的模型可解释性分析页面，不依赖训练页面："),
        bullet("导入Y和X数据（与训练页相同的格式）"),
        bullet("选择树模型类型（RF/XGBoost/LightGBM/GBR/ExtraTrees）并设置参数"),
        bullet("设置训练/预测时间范围"),
        bullet("一键训练+SHAP分析，输出特征重要性排序和可视化图表"),
        p("SHAP值解读：正值表示该特征推高了预测值，负值表示拉低了预测值。mean(|SHAP|)越大说明该特征越重要。"),

        new Paragraph({ spacing: { before: 200 } }),
        h("3.6 逐像元分析", 2),
        p("对单个像素进行时间序列分析："),
        bullet("输入行列坐标，查看该像素完整时间序列（Y值变化曲线）"),
        bullet("叠加所有模型的预测值，训练期和预测期用灰色竖线分隔"),
        bullet("同时展示各X变量的时间序列"),
        bullet("支持随机采样9个像素快速概览"),

        new Paragraph({ children: [new PageBreak()] }),

        h("4. 数据格式要求", 1),
        h("4.1 文件格式", 2),
        bullet("必须为单波段GeoTIFF格式（.tif 或 .tiff）"),
        bullet("所有TIFF文件必须具有相同的空间分辨率、投影坐标系和行列数"),
        bullet("nodata（无效值）会被自动识别并转为NaN处理"),

        h("4.2 文件命名", 2),
        bullet("年度数据：YYYY.tif（如 2014.tif）"),
        bullet("月度数据（平铺）：YYYY-MM.tif（如 2014-01.tif）"),
        bullet("月度数据（嵌套）：YYYY/MM.tif（如 2014/01.tif，每个年份一个子文件夹）"),

        h("4.3 目录结构示例", 2),
        p("年度数据：", { bold: true }),
        p("Y文件夹/                    X1文件夹/"),
        p("  2014.tif                    2014.tif"),
        p("  2015.tif                    2015.tif"),
        p("  ...                         ..."),
        p("  2023.tif                    2023.tif"),
        new Paragraph({ spacing: { before: 120 } }),
        p("月度嵌套数据：", { bold: true }),
        p("Y文件夹/"),
        p("  2014/                       2015/"),
        p("    01.tif                      01.tif"),
        p("    02.tif                      02.tif"),
        p("    ...                         ..."),
        p("    12.tif                      12.tif"),

        new Paragraph({ children: [new PageBreak()] }),
        h("5. 输出文件说明", 1),
        p("训练完成后，在输出目录下生成以下结构："),
        new Paragraph({ spacing: { before: 120 } }),
        // Output structure table
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [2500, 3000, 3860],
          rows: [
            new TableRow({ children: [thc("目录/文件", 2500), thc("内容", 3000), thc("说明", 3860)] }),
            new TableRow({ children: [
              tc("predictions/", 2500, true), tc("预测结果TIFF", 3000), tc("每个模型每个预测年份一张TIFF", 3860)
            ]}),
            new TableRow({ children: [
              tc("charts/", 2500, true), tc("可视化图表PNG", 3000), tc("散点图、残差图、RMSE箱线图、特征重要性等", 3860)
            ], shading: lightBg }),
            new TableRow({ children: [
              tc("tables/", 2500, true), tc("数据表格CSV", 3000), tc("模型评估汇总、逐像素指标、特征重要性", 3860)
            ]}),
            new TableRow({ children: [
              tc("reports/", 2500, true), tc("评估报告", 3000), tc("HTML格式评估报告", 3860)
            ], shading: lightBg }),
            new TableRow({ children: [
              tc("models/", 2500, true), tc("模型持久化文件", 3000), tc("pixel_data.joblib + metadata.json", 3860)
            ]}),
            new TableRow({ children: [
              tc("config.json", 2500, true), tc("运行配置", 3000), tc("本次训练的所有参数记录", 3860)
            ], shading: lightBg }),
          ]
        }),
        new Paragraph({ spacing: { before: 200 } }),
        p("这些文件可用于二次分析、绘图和论文发表。"),

        new Paragraph({ children: [new PageBreak()] }),
        h("6. 常见问题", 1),
        h("6.1 所有模型均未产出有效评估指标", 2),
        bullet("检查训练时间范围是否与数据文件名匹配"),
        bullet("检查TIFF数据中nodata值是否过多 — 使用数据探索页面查看有效像素比例"),
        bullet("确保每个像素至少有2个有效的训练时间步"),
        bullet("如果仅预测1年，R²会显示NaN（需要≥2个预测样本），RMSE仍可正常显示"),
        new Paragraph({ spacing: { before: 120 } }),
        h("6.2 预测单年结果显示为空", 2),
        p("这是正常行为。预测单年时R²无法计算（至少需要2个样本），但RMSE和MAE会正常显示。最佳模型将根据RMSE排序。"),
        new Paragraph({ spacing: { before: 120 } }),
        h("6.3 图表中R²显示为方框", 2),
        p("已在v2.2中修复，改用数学文本渲染 $R^2$。如果仍有问题，请确保系统安装了中文字体（如微软雅黑）。"),
        new Paragraph({ spacing: { before: 120 } }),
        h("6.4 数据读取很慢", 2),
        bullet("TIFF文件大小和数量会影响加载速度"),
        bullet("可以通过减小有效像素比例阈值来减少训练数据量"),
        bullet("月度数据因文件数多，加载时间更长"),
        new Paragraph({ spacing: { before: 120 } }),
        h("6.5 如何反复调用已训练的模型", 2),
        p("训练完成后模型自动保存。使用「模型复用」页面加载任意模型包，配合新X数据即可快速预测，无需重新训练。"),

        new Paragraph({ spacing: { before: 400 }, alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "— 全文完 —", font: "Microsoft YaHei", size: 22, color: "999999", italics: true })] }),
      ]
    }
  ]
});

// Write to file
const outPath = "D:/geospatial_ml_app/地理空间TIFF机器学习平台_使用说明书.docx";
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(outPath, buffer);
  console.log("OK: " + outPath);
});
