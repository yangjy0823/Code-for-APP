import 'dart:math';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:myoelectric_ftg/constant.dart';

class LinesChart extends StatefulWidget {
  final List<List<double?>> yAxisDataArr;
  final List<String> legends;
  final double maxY;
  final String xAxisName;
  final List<int> splitLines;
  final List<String> splitLinesLabels;
  final List<int?> tsList;
  const LinesChart({
    super.key,
    required this.yAxisDataArr,
    required this.maxY,
    required this.legends,
    this.tsList = const [],
    this.xAxisName = '',
    this.splitLines = const [0, 10, 50, 100],
    this.splitLinesLabels = const ['0', '10', '50', '100'],
  }) : assert(
         splitLines.length == splitLinesLabels.length,
       );
  @override
  State<LinesChart> createState() => _LineChartState();
}

class _LineChartState extends State<LinesChart> {
  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        _buildLegends(),
        CustomPaint(
          size: Size(MediaQuery.of(context).size.width, 200),
          painter: _LineChartPainter(
            yAxisDataArr: widget.yAxisDataArr,
            maxY: widget.maxY,
            xAxisName: widget.xAxisName,
            splitLines: widget.splitLines,
            splitLinesLabels: widget.splitLinesLabels,
            tsList: widget.tsList,
          ),
        ),
      ],
    );
  }

  Widget _buildLegends() {
    return Wrap(
      spacing: 8.0, // 主轴方向子组件之间的间距
      runSpacing: 4.0,
      alignment: WrapAlignment.end,
      children: [
        ...widget.legends.asMap().entries.map((entry) {
          final index = entry.key;
          final e = entry.value;
          return Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 10,
                height: 10,
                margin: EdgeInsets.fromLTRB(0, 0, 3, 0),
                decoration: BoxDecoration(
                  color: chartColors[index % chartColors.length],
                  shape: BoxShape.circle,
                ),
              ),
              Text(
                e,
                textAlign: TextAlign.center,
                style: TextStyle(color: progressBkgColor, fontSize: 12),
              ),
            ],
          );
        }),
      ],
    );
  }
}

class _LineChartPainter extends CustomPainter {
  final List<List<double?>> yAxisDataArr;
  final double maxY;
  final String xAxisName;
  final List<int> splitLines;
  final List<String> splitLinesLabels;
  final List<int?> tsList;

  _LineChartPainter({
    required this.yAxisDataArr,
    required this.maxY,
    required this.xAxisName,
    required this.splitLines,
    required this.splitLinesLabels,
    required this.tsList,
  });
  final labelWidth = 40.0;
  final labelHeight = 14.0;
  final xAxisNameWidth = 70.0;

  @override
  void paint(Canvas canvas, Size size) {
    if (yAxisDataArr.isEmpty) return;

    final paintArr = [];
    for (int i = 0; i < yAxisDataArr.length; i++) {
      paintArr.add(
        Paint()
          ..color = chartColors[i % chartColors.length]
          ..strokeWidth = 1.5
          ..isAntiAlias = true
          ..style = PaintingStyle.stroke,
      );
    }

    final chartWidth = size.width - labelWidth * 2;
    final chartHeight = size.height - labelHeight * 2;

    final dashLinePaint = Paint()
      ..color = Colors.red
      ..strokeWidth = 1.0
      ..isAntiAlias = true
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round
      ..style = PaintingStyle.stroke;

    _drawYAxisLabels2(canvas, chartWidth, chartHeight);
    _drawXAxisName(
      canvas,
      chartHeight,
      Offset(chartWidth - xAxisNameWidth, chartHeight + labelHeight * 0.5),
      xAxisName,
      progressBkgColor,
    );

    canvas.save();

    canvas.translate(labelWidth, labelHeight * 0.6);
    canvas.scale(1, -1);
    canvas.translate(0, -chartHeight);

    for (int i = 0; i < splitLines.length; i++) {
      dashLinePaint.color = splitLineColors[i % splitLineColors.length];
      final dy = splitLines[i] * chartHeight / 100;
      _drawDashLine(
        canvas,
        Offset(0, dy),
        Offset(chartWidth, dy),
        dashLinePaint,
      );
    }

    final List<List<List<Offset>>> seriesArr = List.generate(
      yAxisDataArr.length,
      (index) => [],
    );

    final len = yAxisDataArr[0].length;

    calcListOffset(int s, int e, int dataIndex) {
      List<Offset> list = [];
      for (int i = s; i < e; i++) {
        list.add(
          Offset(
            // (tsList[i]! - startTs) * chartWidth / timeLen,
            (i) * chartWidth / len,
            yAxisDataArr[dataIndex][i]! * chartHeight / maxY,
          ),
        );
      }
      return list;
    }

    for (int i = 0; i < yAxisDataArr.length; i++) {
      int s = 0;
      int e = 0;
      s = yAxisDataArr[i].indexWhere((n) => n != null);
      e = s;
      if (s != -1) {
        while (s < yAxisDataArr[i].length && e < yAxisDataArr[i].length) {
          if (yAxisDataArr[i][e] != null) {
            e++;
          } else {
            seriesArr[i].add(calcListOffset(s, e, i));
            s = e;
            s = yAxisDataArr[i].indexWhere((n) => n != null, s);
            e = s;
            if (s == -1) {
              break;
            }
          }
        }
      }
      if (yAxisDataArr[i].isNotEmpty && yAxisDataArr[i].last != null) {
        seriesArr[i].add(calcListOffset(s, yAxisDataArr[i].length, i));
      }
      for (int j = 0; j < seriesArr[i].length; j++) {
        canvas.drawPoints(ui.PointMode.polygon, seriesArr[i][j], paintArr[i]);
      }
    }

    canvas.restore();
  }

  _drawDashLine(
    Canvas canvas,
    Offset startOffset,
    Offset endOffset,
    Paint paint,
  ) {
    final List<double> dashPattern = [10, 5];
    const dashWidth = 6.0;
    final gapWidth = dashWidth * dashPattern[1] / dashPattern[0];
    final path = Path();
    double p = startOffset.dx;
    while (p <= endOffset.dx) {
      path.moveTo(p, startOffset.dy);
      p = p + dashWidth > endOffset.dx ? endOffset.dx : (p + dashWidth);
      path.lineTo(p, endOffset.dy);
      p += gapWidth;
    }
    canvas.drawPath(path, paint);
  }

  _drawYAxisLabels(Canvas canvas, double chartWidth, double chartHeight) {
    _drawYLabel(
      canvas,
      chartHeight,
      Offset(0, chartHeight),
      '0',
      progressBkgColor,
    );
    _drawYLabel(
      canvas,
      chartHeight,
      Offset(0, chartHeight * (1 - 0.1)),
      '10',
      progressPhase1Color,
    );
    _drawYLabel(
      canvas,
      chartHeight,
      Offset(0, chartHeight * 0.5),
      '50',
      progressPhase2Color,
    );
    _drawYLabel(
      canvas,
      chartHeight,
      const Offset(0, 0),
      '100',
      progressPhase3Color,
    );
  }

  _drawYAxisLabels2(Canvas canvas, double chartWidth, double chartHeight) {
    for (int i = 0; i < splitLines.length; i++) {
      final percent = splitLines[i] / 100;
      _drawYLabel(
        canvas,
        chartHeight,
        Offset(0, chartHeight * (1 - percent)),
        splitLinesLabels[i],
        splitLineColors[i % splitLineColors.length],
      );
    }
  }

  _drawYLabel(
    Canvas canvas,
    double chartHeight,
    Offset offset,
    String str,
    Color color,
  ) {
    final paragraphBuilderY3 =
        ui.ParagraphBuilder(
            ui.ParagraphStyle(
              textAlign: TextAlign.center,
              fontSize: 5,
              height: 0,
            ),
          )
          ..pushStyle(ui.TextStyle(color: color, fontSize: labelHeight))
          ..addText(str);
    final paragraphY3 = paragraphBuilderY3.build();
    paragraphY3.layout(ui.ParagraphConstraints(width: labelWidth));
    canvas.drawParagraph(paragraphY3, offset);
  }

  _drawXAxisName(
    Canvas canvas,
    double chartHeight,
    Offset offset,
    String str,
    Color color,
  ) {
    final paragraphBuilderY3 =
        ui.ParagraphBuilder(
            ui.ParagraphStyle(
              textAlign: TextAlign.right,
              fontSize: 6,
              height: 0,
            ),
          )
          ..pushStyle(ui.TextStyle(color: color, fontSize: labelHeight))
          ..addText(str);
    final paragraphY3 = paragraphBuilderY3.build();
    paragraphY3.layout(ui.ParagraphConstraints(width: xAxisNameWidth * 1.6));
    canvas.drawParagraph(paragraphY3, offset);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }

}