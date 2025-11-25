import 'package:flutter/material.dart';

class ArrowChart extends StatelessWidget {
  final List<List<int>> data;
  final List<String> yAxisLabels;
  final List<String> xAxisLabels;
  final double width;

  final Color labelColor;
  final String xAxisName;
  final double xAxisNameWidth;
  final double gapBetweenQuenes;
  final double yAxisLabelWidth;
  final double gapBetweenLabelsAndYAxis;
  final double queneHeight;
  final double gap;
  final int xAxisMode;
  const ArrowChart({
    super.key,
    required this.data,
    required this.yAxisLabels,
    required this.xAxisLabels,
    required this.width,
    this.labelColor = Colors.black45,
    this.xAxisName = "",
    this.xAxisNameWidth = 80,
    this.gapBetweenQuenes = 10,
    this.yAxisLabelWidth = 110,
    this.gapBetweenLabelsAndYAxis = 5,
    this.queneHeight = 44,
    this.gap = 1.0,
    this.xAxisMode = 0,
  });

  double get queneWidth =>
      width - yAxisLabelWidth - gapBetweenLabelsAndYAxis - xAxisNameWidth;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [...buildQuenes(), buildXAxisLine(), buildXAxisLabels()],
    );
  }

  List<Widget> buildQuenes() {
    List<Widget> quenes = [];
    for (int i = 0; i < data.length; i++) {
      final queue = data[i];
      final row = Row(
        children: [
          SizedBox(
            width: yAxisLabelWidth,
            child: Text(
              yAxisLabels[i],
              textAlign: TextAlign.right,
              style: TextStyle(color: labelColor,fontSize: 16.0),
            ),
          ),
          SizedBox(width: gapBetweenLabelsAndYAxis),
          CustomPaint(
            size: Size(queneWidth, queneHeight),
            painter: ArrowQuenePainter(quene: queue, gap: gap),
          ),
        ],
      );
      quenes.add(row);
      quenes.add(SizedBox(height: gapBetweenQuenes));
    }
    return quenes;
  }

  Widget buildXAxisLine() {
    return Row(
      children: [
        SizedBox(width: yAxisLabelWidth),
        SizedBox(width: gapBetweenLabelsAndYAxis),
        CustomPaint(
          size: Size(queneWidth + xAxisNameWidth / 2, 4),
          painter: XAxisLinePainter(color: labelColor),
        ),
      ],
    );
  }

  Widget buildXAxisLabels() {
    final cnt = data[0].length;
    final w = queneWidth;
    final h = queneHeight;

    final triW = h / 2;
    final topW = (w - triW - gap * (cnt - 1)) / cnt;
    if (xAxisMode == 0) {
      List<Widget> positionedArr = [];
      for (int i = 0; i < xAxisLabels.length; i++) {
        final p = Positioned(
          left: (i + 1) * (topW + gap),
          child: Text("${xAxisLabels[i]}", style: TextStyle(color: labelColor)),
        );
        positionedArr.add(p);
      }

      return Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          SizedBox(width: yAxisLabelWidth),
          SizedBox(
            width: queneWidth,
            height: 30,
            child: Stack(children: positionedArr),
          ),
          SizedBox(
            width: xAxisNameWidth,
            height: 30,
            child: Text(
              xAxisName,
              textAlign: TextAlign.center,
              style: TextStyle(color: labelColor),
            ),
          ),
        ],
      );
    } else {
      return Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          SizedBox(width: yAxisLabelWidth),
          Expanded(child: Container()),
          SizedBox(
            width: xAxisNameWidth * 3,
            height: 30,
            child: Text(
              xAxisName,
              textAlign: TextAlign.center,
              style: TextStyle(color: labelColor),
            ),
          ),
        ],
      );
    }
  }
}

class ArrowQuenePainter extends CustomPainter {
  final List<int> quene;
  final double gap;
  ArrowQuenePainter({required this.quene, required this.gap})
    : assert(quene.isNotEmpty);

  @override
  void paint(Canvas canvas, Size size) {
    final cnt = quene.length;
    final h = size.height;
    final w = size.width;

    final triW = h / 2;
    final topW = (w - triW - gap * (cnt - 1)) / cnt;

    final colors = [
      Color.fromARGB(255, 146, 208, 80),
      Color.fromARGB(255, 255, 192, 0),
      Color.fromARGB(255, 255, 0, 0),
      Colors.transparent,
    ];

    var paint = Paint()
      ..isAntiAlias = true
      ..strokeWidth = 1
      ..style = PaintingStyle.fill
      ..color = Colors.red;

    for (int i = 0; i < cnt; i++) {
      if (quene[i] < 0) {
        paint.color = Colors.transparent;
      } else {
        paint.color = colors[quene[i]];
      }
      if (i == 0) {
        canvas.drawPath(
          Path()
            ..moveTo(0 + i * topW + gap * i, 0)
            ..lineTo((i + 1) * topW + gap * i, 0)
            ..lineTo((i + 1) * topW + triW + gap * i, triW)
            ..lineTo((i + 1) * topW + gap * i, h)
            ..lineTo(i * topW + gap * i, h)
            ..close(),
          paint,
        );
      } else {
        canvas.drawPath(
          Path()
            ..moveTo(0 + i * topW + gap * i, 0)
            ..lineTo((i + 1) * topW + gap * i, 0)
            ..lineTo((i + 1) * topW + triW + gap * i, triW)
            ..lineTo((i + 1) * topW + gap * i, h)
            ..lineTo(i * topW + gap * i, h)
            ..lineTo(i * topW + triW + gap * i, triW)
            ..close(),
          paint,
        );
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return false;
  }

  drawBorder(Canvas canvas, Size size) {
    var rect = Offset.zero & size;
    var paint = Paint()
      ..isAntiAlias = true
      ..strokeWidth = 1
      ..style = PaintingStyle.stroke
      ..color = Colors.grey;
    canvas.drawRect(rect, paint);
  }
}

class XAxisLinePainter extends CustomPainter {
  const XAxisLinePainter({this.color = Colors.black45});

  final Color color;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeWidth = 1
      ..style = PaintingStyle.stroke;

    final double arrowSize = 6.0;
    final double lineHeight = size.height / 2;

    canvas.drawLine(
      Offset(0, lineHeight),
      Offset(size.width, lineHeight),
      paint,
    );

    final Path arrowPath = Path()
      ..moveTo(size.width - arrowSize, lineHeight - arrowSize * 2 / 3)
      ..lineTo(size.width, lineHeight)
      ..lineTo(size.width - arrowSize, lineHeight + arrowSize * 2 / 3);

    canvas.drawPath(arrowPath, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }
}