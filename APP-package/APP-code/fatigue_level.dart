import 'dart:math';

import 'package:flutter/material.dart';
import 'package:myoelectric_ftg/constant.dart';

class FatigueLevel extends StatelessWidget {
  final double value;
  final double width;

  const FatigueLevel({super.key, required this.value, this.width = 300})
    : assert(value >= 0 && value <= 2);

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text(
          "Fatigue Level",
          style: TextStyle(fontSize: 16, color: progressBkgColor),
        ),
        SizedBox(height: 18),
        CustomPaint(
          size: Size(width, width / 2),
          painter: FatigueLevelPainter(value: value),
        ),
        SizedBox(height: 40),
      ],
    );
  }
}

class FatigueLevelPainter extends CustomPainter {
  final double value;
  FatigueLevelPainter({required this.value}) : assert(value >= 0 && value <= 2);

  @override
  void paint(Canvas canvas, Size size) {
    final barWidth = 20.0;
    final rect = Rect.fromLTWH(0, 0, size.width, size.width);
    final colors = [
      Color.fromARGB(255, 146, 208, 80),
      Color.fromARGB(255, 255, 192, 0),
      Color.fromARGB(255, 255, 0, 0),
    ];

    final arcBarPaint = Paint()
      ..isAntiAlias = true
      ..strokeWidth = barWidth
      ..style = PaintingStyle.stroke
      ..color = colors[0];

    canvas.drawArc(rect, pi, pi / 3, false, arcBarPaint);
    arcBarPaint.color = colors[1];
    canvas.drawArc(rect, pi + pi / 3, pi / 3, false, arcBarPaint);
    arcBarPaint.color = colors[2];
    canvas.drawArc(rect, pi + pi / 3 * 2, pi / 3, false, arcBarPaint);

    final capPaint = Paint()
      ..isAntiAlias = true
      ..strokeWidth = barWidth
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;
    canvas.drawArc(rect, pi, 0.01, false, capPaint..color = colors[0]);
    canvas.drawArc(
      rect,
      pi + pi - 0.01,
      0.01,
      false,
      capPaint..color = colors[2],
    );

    final progress = value / 2.0;
    final progressAnglge = progress * pi;
    final ox = rect.center.dx;
    final oy = rect.center.dy;
    final h = rect.height / 2.0 - barWidth;
    final bottomBorderWidth = 10.0;
    final bbwHalf = bottomBorderWidth / 2.0;
    final alpha = pi - progressAnglge;
    final x1 = ox + h * cos(alpha);
    final y1 = oy - h * sin(alpha);
    final x2 = ox + bbwHalf * sin(alpha);
    final y2 = oy + bbwHalf * cos(alpha);
    final x3 = ox - bbwHalf * sin(alpha);
    final y3 = oy - bbwHalf * cos(alpha);

    final pointerPaint = Paint()
      ..isAntiAlias = true
      ..color = Colors.black
      ..style = PaintingStyle.fill;

    canvas.drawPath(
      Path()
        ..moveTo(x1, y1)
        ..lineTo(x2, y2)
        ..lineTo(x3, y3)
        ..close(),
      pointerPaint,
    );

    canvas.drawCircle(Offset(ox, oy), 1.3 * bbwHalf, pointerPaint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }
}