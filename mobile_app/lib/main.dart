import 'dart:async';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);
  runApp(const QEdgeApp());
}

class QEdgeApp extends StatelessWidget {
  const QEdgeApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Q-Edge',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF0F1419),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF5B8DEF),
          secondary: Color(0xFF7C5CFF),
          surface: Color(0xFF1A1F2E),
        ),
        fontFamily: 'SF Pro Text',
      ),
      home: const QEdgeDashboard(),
    );
  }
}

class QEdgeDashboard extends StatefulWidget {
  const QEdgeDashboard({Key? key}) : super(key: key);

  @override
  State<QEdgeDashboard> createState() => _QEdgeDashboardState();
}

class _QEdgeDashboardState extends State<QEdgeDashboard>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;

  bool _isTraining = false;
  bool _isPQCConnected = false;
  int _currentRound = 0;
  final int _totalRounds = 10;
  double _currentLoss = 0.0;
  double _currentAccuracy = 0.0;

  final List<double> _lossHistory = [];
  final List<double> _accuracyHistory = [];
  final List<PQCLogEntry> _pqcLogs = [];
  final List<QuantumState> _quantumStates = [];

  Timer? _trainingTimer;
  Timer? _quantumTimer;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 4, vsync: this);
    _initializePQC();
    _initializeQuantumSimulation();
  }

  @override
  void dispose() {
    _tabController.dispose();
    _trainingTimer?.cancel();
    _quantumTimer?.cancel();
    super.dispose();
  }

  void _initializePQC() {
    Future.delayed(const Duration(seconds: 1), () {
      _addPQCLog('Generating Kyber-1024 keypair...', LogType.info);
    });

    Future.delayed(const Duration(seconds: 2), () {
      _addPQCLog('Keypair generated successfully', LogType.success);
      _addPQCLog('Initiating key encapsulation...', LogType.info);
    });

    Future.delayed(const Duration(seconds: 3), () {
      _addPQCLog('Shared secret established', LogType.success);
      _addPQCLog('Generating Dilithium-5 signature...', LogType.info);
    });

    Future.delayed(const Duration(seconds: 4), () {
      _addPQCLog('PQC tunnel established', LogType.success);
      setState(() => _isPQCConnected = true);
    });
  }

  void _initializeQuantumSimulation() {
    _quantumTimer = Timer.periodic(const Duration(milliseconds: 800), (timer) {
      if (_quantumStates.length >= 50) {
        _quantumStates.removeAt(0);
      }

      final random = Random();
      setState(() {
        _quantumStates.add(
          QuantumState(
            qubit0: random.nextDouble(),
            qubit1: random.nextDouble(),
            entanglement: 0.5 + 0.5 * sin(timer.tick * 0.1),
            coherence: 0.8 + 0.2 * cos(timer.tick * 0.15),
            timestamp: DateTime.now(),
          ),
        );
      });
    });
  }

  void _addPQCLog(String message, LogType type) {
    setState(() {
      _pqcLogs.add(
        PQCLogEntry(message: message, type: type, timestamp: DateTime.now()),
      );
    });
  }

  void _startTraining() {
    if (_isTraining || !_isPQCConnected) return;

    setState(() {
      _isTraining = true;
      _currentRound = 0;
      _lossHistory.clear();
      _accuracyHistory.clear();
    });

    _addPQCLog('Starting federated learning...', LogType.info);

    _trainingTimer = Timer.periodic(const Duration(seconds: 2), (timer) {
      if (_currentRound >= _totalRounds) {
        timer.cancel();
        setState(() => _isTraining = false);
        _addPQCLog('Training complete', LogType.success);
        return;
      }

      final random = Random();
      final newLoss =
          2.0 * exp(-0.3 * _currentRound) + random.nextDouble() * 0.1;
      final newAccuracy =
          min(0.98, 0.5 + 0.05 * _currentRound + random.nextDouble() * 0.02);

      setState(() {
        _currentRound++;
        _currentLoss = newLoss;
        _currentAccuracy = newAccuracy;
        _lossHistory.add(newLoss);
        _accuracyHistory.add(newAccuracy);
      });

      _addPQCLog('Round $_currentRound completed', LogType.info);

      if (_currentRound % 3 == 0) {
        _addPQCLog('Quantum aggregation triggered', LogType.quantum);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0F1419),
      body: SafeArea(
        child: Column(
          children: [
            _buildHeader(),
            _buildMetricsRow(),
            _buildTabBar(),
            Expanded(
              child: TabBarView(
                controller: _tabController,
                children: [
                  _buildTrainingTab(),
                  _buildQuantumTab(),
                  _buildSecurityTab(),
                  _buildSettingsTab(),
                ],
              ),
            ),
          ],
        ),
      ),
      floatingActionButton: _buildFAB(),
    );
  }

  Widget _buildHeader() {
    return Container(
      padding: const EdgeInsets.fromLTRB(20, 16, 20, 12),
      child: Row(
        children: [
          Container(
            width: 40,
            height: 40,
            decoration: BoxDecoration(
              color: const Color(0xFF5B8DEF).withOpacity(0.15),
              borderRadius: BorderRadius.circular(10),
            ),
            child: const Center(
              child: Text(
                'Q',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w700,
                  color: Color(0xFF5B8DEF),
                ),
              ),
            ),
          ),
          const SizedBox(width: 12),
          const Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Q-Edge',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w600,
                  color: Colors.white,
                  letterSpacing: -0.3,
                ),
              ),
              Text(
                'Federated Quantum Neural Network',
                style: TextStyle(
                  fontSize: 11,
                  color: Color(0xFF6B7280),
                ),
              ),
            ],
          ),
          const Spacer(),
          _buildConnectionBadge(),
        ],
      ),
    );
  }

  Widget _buildConnectionBadge() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: _isPQCConnected
            ? const Color(0xFF059669).withOpacity(0.12)
            : const Color(0xFFD97706).withOpacity(0.12),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(
          color: _isPQCConnected
              ? const Color(0xFF059669).withOpacity(0.3)
              : const Color(0xFFD97706).withOpacity(0.3),
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 6,
            height: 6,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: _isPQCConnected
                  ? const Color(0xFF10B981)
                  : const Color(0xFFF59E0B),
            ),
          ),
          const SizedBox(width: 6),
          Text(
            _isPQCConnected ? 'Secured' : 'Connecting',
            style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w500,
              color: _isPQCConnected
                  ? const Color(0xFF10B981)
                  : const Color(0xFFF59E0B),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMetricsRow() {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(
        children: [
          Expanded(
              child: _buildMetricCard('Round', '$_currentRound/$_totalRounds',
                  const Color(0xFF5B8DEF))),
          const SizedBox(width: 8),
          Expanded(
              child: _buildMetricCard('Loss', _currentLoss.toStringAsFixed(3),
                  const Color(0xFFD97706))),
          const SizedBox(width: 8),
          Expanded(
              child: _buildMetricCard(
                  'Accuracy',
                  '${(_currentAccuracy * 100).toStringAsFixed(1)}%',
                  const Color(0xFF059669))),
          const SizedBox(width: 8),
          Expanded(
              child: _buildMetricCard('Status', _isTraining ? 'Active' : 'Idle',
                  const Color(0xFF7C5CFF))),
        ],
      ),
    );
  }

  Widget _buildMetricCard(String label, String value, Color accentColor) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 10),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1F2E),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: const Color(0xFF2D3748)),
      ),
      child: Column(
        children: [
          Text(
            value,
            style: TextStyle(
              fontSize: 15,
              fontWeight: FontWeight.w600,
              color: accentColor,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            label,
            style: const TextStyle(
              fontSize: 10,
              color: Color(0xFF6B7280),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTabBar() {
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 8, 16, 4),
      height: 40,
      decoration: BoxDecoration(
        color: const Color(0xFF1A1F2E),
        borderRadius: BorderRadius.circular(8),
      ),
      child: TabBar(
        controller: _tabController,
        indicator: BoxDecoration(
          color: const Color(0xFF2D3748),
          borderRadius: BorderRadius.circular(6),
        ),
        indicatorSize: TabBarIndicatorSize.tab,
        indicatorPadding: const EdgeInsets.all(3),
        labelColor: Colors.white,
        unselectedLabelColor: const Color(0xFF6B7280),
        labelStyle: const TextStyle(fontSize: 12, fontWeight: FontWeight.w500),
        dividerColor: Colors.transparent,
        tabs: const [
          Tab(text: 'Training'),
          Tab(text: 'Quantum'),
          Tab(text: 'Security'),
          Tab(text: 'Settings'),
        ],
      ),
    );
  }

  Widget _buildTrainingTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildSectionLabel('Loss Curve'),
          const SizedBox(height: 8),
          _buildChartCard(_lossHistory, const Color(0xFFD97706), 2.5),
          const SizedBox(height: 20),
          _buildSectionLabel('Accuracy Curve'),
          const SizedBox(height: 8),
          _buildChartCard(_accuracyHistory, const Color(0xFF059669), 1.0),
          const SizedBox(height: 20),
          _buildSectionLabel('Configuration'),
          const SizedBox(height: 8),
          _buildConfigSection(),
          const SizedBox(height: 80),
        ],
      ),
    );
  }

  Widget _buildSectionLabel(String title) {
    return Text(
      title,
      style: const TextStyle(
        fontSize: 13,
        fontWeight: FontWeight.w600,
        color: Color(0xFF9CA3AF),
        letterSpacing: 0.3,
      ),
    );
  }

  Widget _buildChartCard(List<double> data, Color color, double maxY) {
    return Container(
      height: 160,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1F2E),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFF2D3748)),
      ),
      child: CustomPaint(
        size: const Size(double.infinity, 128),
        painter: SimpleChartPainter(data: data, color: color, maxY: maxY),
      ),
    );
  }

  Widget _buildConfigSection() {
    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF1A1F2E),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFF2D3748)),
      ),
      child: Column(
        children: [
          _buildConfigRow('Local Epochs', '5', true),
          _buildConfigRow('Batch Size', '32', false),
          _buildConfigRow('Learning Rate', '0.001', false),
          _buildConfigRow('Compression', '10%', false),
          _buildConfigRow('Aggregation', 'FedAvg + VQC', false),
        ],
      ),
    );
  }

  Widget _buildConfigRow(String label, String value, bool isFirst) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      decoration: BoxDecoration(
        border: isFirst
            ? null
            : const Border(
                top: BorderSide(color: Color(0xFF2D3748)),
              ),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label,
              style: const TextStyle(color: Color(0xFF9CA3AF), fontSize: 13)),
          Text(value,
              style: const TextStyle(
                  color: Colors.white,
                  fontSize: 13,
                  fontWeight: FontWeight.w500)),
        ],
      ),
    );
  }

  Widget _buildQuantumTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildSectionLabel('Entanglement Status'),
          const SizedBox(height: 8),
          _buildEntanglementSection(),
          const SizedBox(height: 20),
          _buildSectionLabel('Qubit Amplitudes'),
          const SizedBox(height: 8),
          _buildQubitSection(),
          const SizedBox(height: 20),
          _buildSectionLabel('VQC Parameters'),
          const SizedBox(height: 8),
          _buildVQCSection(),
          const SizedBox(height: 80),
        ],
      ),
    );
  }

  Widget _buildEntanglementSection() {
    final latestState =
        _quantumStates.isNotEmpty ? _quantumStates.last : QuantumState();

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1F2E),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFF2D3748)),
      ),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              _buildQubitIndicator(
                  'Q0', latestState.qubit0, const Color(0xFF5B8DEF)),
              Container(
                width: 60,
                height: 2,
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [
                      const Color(0xFF5B8DEF).withOpacity(0.3),
                      const Color(0xFF7C5CFF)
                          .withOpacity(0.8 * latestState.entanglement),
                      const Color(0xFF7C5CFF).withOpacity(0.3),
                    ],
                  ),
                  borderRadius: BorderRadius.circular(1),
                ),
              ),
              _buildQubitIndicator(
                  'Q1', latestState.qubit1, const Color(0xFF7C5CFF)),
            ],
          ),
          const SizedBox(height: 20),
          Row(
            children: [
              Expanded(
                  child: _buildStatBox('Entanglement',
                      '${(latestState.entanglement * 100).toStringAsFixed(0)}%')),
              const SizedBox(width: 10),
              Expanded(
                  child: _buildStatBox('Coherence',
                      '${(latestState.coherence * 100).toStringAsFixed(0)}%')),
              const SizedBox(width: 10),
              Expanded(
                  child: _buildStatBox('Fidelity',
                      '${(95 + 5 * latestState.coherence).toStringAsFixed(0)}%')),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildQubitIndicator(String label, double value, Color color) {
    return Column(
      children: [
        Container(
          width: 56,
          height: 56,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: color.withOpacity(0.1),
            border: Border.all(color: color.withOpacity(0.4), width: 2),
          ),
          child: Center(
            child: Text(
              '|${value > 0.5 ? "1" : "0"}⟩',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w600,
                color: color,
              ),
            ),
          ),
        ),
        const SizedBox(height: 8),
        Text(label,
            style: const TextStyle(color: Color(0xFF6B7280), fontSize: 12)),
      ],
    );
  }

  Widget _buildStatBox(String label, String value) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 12),
      decoration: BoxDecoration(
        color: const Color(0xFF0F1419),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        children: [
          Text(
            value,
            style: const TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w600,
              color: Colors.white,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            label,
            style: const TextStyle(
              fontSize: 10,
              color: Color(0xFF6B7280),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildQubitSection() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1F2E),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFF2D3748)),
      ),
      child: Column(
        children: List.generate(8, (index) {
          final random = Random(index + DateTime.now().second);
          final amplitude = 0.3 + random.nextDouble() * 0.7;
          final color = Color.lerp(
              const Color(0xFF5B8DEF), const Color(0xFF7C5CFF), index / 7)!;

          return Padding(
            padding: const EdgeInsets.symmetric(vertical: 6),
            child: Row(
              children: [
                SizedBox(
                  width: 28,
                  child: Text(
                    'q$index',
                    style: const TextStyle(
                      fontFamily: 'monospace',
                      color: Color(0xFF6B7280),
                      fontSize: 12,
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Container(
                    height: 6,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(3),
                      color: const Color(0xFF2D3748),
                    ),
                    child: FractionallySizedBox(
                      alignment: Alignment.centerLeft,
                      widthFactor: amplitude,
                      child: Container(
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(3),
                          color: color,
                        ),
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                SizedBox(
                  width: 36,
                  child: Text(
                    amplitude.toStringAsFixed(2),
                    style: TextStyle(
                      fontFamily: 'monospace',
                      color: color,
                      fontSize: 11,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ),
              ],
            ),
          );
        }),
      ),
    );
  }

  Widget _buildVQCSection() {
    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF1A1F2E),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFF2D3748)),
      ),
      child: Column(
        children: [
          _buildConfigRow('Qubits', '8', true),
          _buildConfigRow('Layers', '4', false),
          _buildConfigRow('Ansatz', 'StronglyEntangling', false),
          _buildConfigRow('Parameters', '96', false),
          _buildConfigRow('Circuit Depth', '17', false),
        ],
      ),
    );
  }

  Widget _buildSecurityTab() {
    return Column(
      children: [
        _buildSecurityStatus(),
        Expanded(
          child: ListView.builder(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
            itemCount: _pqcLogs.length,
            itemBuilder: (context, index) {
              final log = _pqcLogs[_pqcLogs.length - 1 - index];
              return _buildLogItem(log);
            },
          ),
        ),
      ],
    );
  }

  Widget _buildSecurityStatus() {
    return Container(
      margin: const EdgeInsets.all(16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: _isPQCConnected
            ? const Color(0xFF059669).withOpacity(0.08)
            : const Color(0xFFD97706).withOpacity(0.08),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: _isPQCConnected
              ? const Color(0xFF059669).withOpacity(0.2)
              : const Color(0xFFD97706).withOpacity(0.2),
        ),
      ),
      child: Row(
        children: [
          Icon(
            _isPQCConnected
                ? Icons.verified_user_outlined
                : Icons.security_outlined,
            color: _isPQCConnected
                ? const Color(0xFF10B981)
                : const Color(0xFFF59E0B),
            size: 24,
          ),
          const SizedBox(width: 12),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                _isPQCConnected
                    ? 'Post-Quantum Secured'
                    : 'Establishing Connection',
                style: TextStyle(
                  fontWeight: FontWeight.w600,
                  fontSize: 14,
                  color: _isPQCConnected
                      ? const Color(0xFF10B981)
                      : const Color(0xFFF59E0B),
                ),
              ),
              Text(
                _isPQCConnected
                    ? 'Kyber-1024 + Dilithium-5'
                    : 'Initializing...',
                style: const TextStyle(
                  fontSize: 11,
                  color: Color(0xFF6B7280),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildLogItem(PQCLogEntry log) {
    final color = _getLogColor(log.type);

    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1F2E),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFF2D3748)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 6,
            height: 6,
            margin: const EdgeInsets.only(top: 5),
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: color,
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  log.message,
                  style:
                      const TextStyle(color: Color(0xFFE5E7EB), fontSize: 13),
                ),
                const SizedBox(height: 3),
                Text(
                  _formatTime(log.timestamp),
                  style:
                      const TextStyle(fontSize: 10, color: Color(0xFF6B7280)),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSettingsTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildSectionLabel('Server'),
          const SizedBox(height: 8),
          _buildSettingsGroup([
            _buildSettingRow('Server Address', 'api.qedge.ai:443', true),
            _buildSettingRow('Workspace', 'qedge-production', false),
            _buildSettingRow('Connection', 'WebSocket + TLS 1.3', false),
          ]),
          const SizedBox(height: 20),
          _buildSectionLabel('Privacy'),
          const SizedBox(height: 8),
          _buildSettingsGroup([
            _buildToggleRow('Differential Privacy', true, true),
            _buildToggleRow('Local Training Only', true, false),
            _buildToggleRow('Battery-Aware', false, false),
          ]),
          const SizedBox(height: 20),
          _buildSectionLabel('About'),
          const SizedBox(height: 8),
          _buildSettingsGroup([
            _buildSettingRow('Version', '1.0.0-beta', true),
            _buildSettingRow('Client ID', 'mobile_001', false),
            _buildSettingRow('Author', 'Ahmad Rasidi', false),
          ]),
          const SizedBox(height: 80),
        ],
      ),
    );
  }

  Widget _buildSettingsGroup(List<Widget> children) {
    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF1A1F2E),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFF2D3748)),
      ),
      child: Column(children: children),
    );
  }

  Widget _buildSettingRow(String label, String value, bool isFirst) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      decoration: BoxDecoration(
        border: isFirst
            ? null
            : const Border(
                top: BorderSide(color: Color(0xFF2D3748)),
              ),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label,
              style: const TextStyle(color: Color(0xFFE5E7EB), fontSize: 13)),
          Text(value,
              style: const TextStyle(color: Color(0xFF6B7280), fontSize: 13)),
        ],
      ),
    );
  }

  Widget _buildToggleRow(String label, bool value, bool isFirst) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      decoration: BoxDecoration(
        border: isFirst
            ? null
            : const Border(
                top: BorderSide(color: Color(0xFF2D3748)),
              ),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label,
              style: const TextStyle(color: Color(0xFFE5E7EB), fontSize: 13)),
          Switch(
            value: value,
            onChanged: (_) {},
            activeColor: const Color(0xFF5B8DEF),
            activeTrackColor: const Color(0xFF5B8DEF).withOpacity(0.3),
            inactiveThumbColor: const Color(0xFF6B7280),
            inactiveTrackColor: const Color(0xFF2D3748),
          ),
        ],
      ),
    );
  }

  Widget _buildFAB() {
    return FloatingActionButton.extended(
      onPressed: _isPQCConnected ? (_isTraining ? null : _startTraining) : null,
      backgroundColor: _isPQCConnected
          ? (_isTraining ? const Color(0xFF6B7280) : const Color(0xFF5B8DEF))
          : const Color(0xFF374151),
      elevation: 0,
      icon: Icon(
        _isTraining ? Icons.hourglass_top : Icons.play_arrow,
        color: Colors.white,
        size: 20,
      ),
      label: Text(
        _isTraining ? 'Training...' : 'Start Training',
        style: const TextStyle(
          color: Colors.white,
          fontWeight: FontWeight.w500,
          fontSize: 13,
        ),
      ),
    );
  }

  Color _getLogColor(LogType type) {
    switch (type) {
      case LogType.info:
        return const Color(0xFF5B8DEF);
      case LogType.success:
        return const Color(0xFF10B981);
      case LogType.warning:
        return const Color(0xFFF59E0B);
      case LogType.error:
        return const Color(0xFFEF4444);
      case LogType.quantum:
        return const Color(0xFF7C5CFF);
    }
  }

  String _formatTime(DateTime time) {
    return '${time.hour.toString().padLeft(2, '0')}:'
        '${time.minute.toString().padLeft(2, '0')}:'
        '${time.second.toString().padLeft(2, '0')}';
  }
}

class SimpleChartPainter extends CustomPainter {
  final List<double> data;
  final Color color;
  final double maxY;

  SimpleChartPainter(
      {required this.data, required this.color, required this.maxY});

  @override
  void paint(Canvas canvas, Size size) {
    if (data.isEmpty) {
      final textPainter = TextPainter(
        text: TextSpan(
          text: 'Waiting for data...',
          style: TextStyle(
              color: const Color(0xFF6B7280).withOpacity(0.5), fontSize: 13),
        ),
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();
      textPainter.paint(
        canvas,
        Offset((size.width - textPainter.width) / 2,
            (size.height - textPainter.height) / 2),
      );
      return;
    }

    final linePaint = Paint()
      ..color = color
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;

    final fillPaint = Paint()
      ..shader = LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: [color.withOpacity(0.2), color.withOpacity(0.0)],
      ).createShader(Rect.fromLTWH(0, 0, size.width, size.height));

    final path = Path();
    final fillPath = Path();

    for (int i = 0; i < data.length; i++) {
      final x = (i / (data.length - 1).clamp(1, double.infinity)) * size.width;
      final y = size.height - (data[i] / maxY).clamp(0.0, 1.0) * size.height;

      if (i == 0) {
        path.moveTo(x, y);
        fillPath.moveTo(x, size.height);
        fillPath.lineTo(x, y);
      } else {
        path.lineTo(x, y);
        fillPath.lineTo(x, y);
      }
    }

    fillPath.lineTo(size.width, size.height);
    fillPath.close();

    canvas.drawPath(fillPath, fillPaint);
    canvas.drawPath(path, linePaint);

    // Draw last point
    if (data.isNotEmpty) {
      final lastX = size.width;
      final lastY =
          size.height - (data.last / maxY).clamp(0.0, 1.0) * size.height;
      canvas.drawCircle(Offset(lastX, lastY), 4, Paint()..color = color);
    }
  }

  @override
  bool shouldRepaint(covariant SimpleChartPainter oldDelegate) => true;
}

enum LogType { info, success, warning, error, quantum }

class PQCLogEntry {
  final String message;
  final LogType type;
  final DateTime timestamp;

  PQCLogEntry(
      {required this.message, required this.type, required this.timestamp});
}

class QuantumState {
  final double qubit0;
  final double qubit1;
  final double entanglement;
  final double coherence;
  final DateTime timestamp;

  QuantumState({
    this.qubit0 = 0.5,
    this.qubit1 = 0.5,
    this.entanglement = 0.5,
    this.coherence = 0.8,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();
}
