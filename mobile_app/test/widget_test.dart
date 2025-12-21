import 'package:flutter_test/flutter_test.dart';

import 'package:q_edge_mobile/main.dart';

void main() {
  testWidgets('Q-Edge app smoke test', (WidgetTester tester) async {
    await tester.pumpWidget(const QEdgeApp());

    expect(find.text('Q-Edge'), findsOneWidget);
  });
}
