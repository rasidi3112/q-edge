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
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF6C63FF),
          brightness: Brightness.dark,
        ),
        fontFamily: 'Inter',
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
    with TickerProviderStateMixin {
  late TabController _tabController;
  late AnimationController _pulseController;

  bool _isTraining = false;
  bool _isPQCConnected = false;
  int _currentRound = 0;
  int _totalRounds = 10;
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
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat(reverse: true);

    _initializePQC();
    _initializeQuantumSimulation();
  }

  @override
  void dispose() {
    _tabController.dispose();
    _pulseController.dispose();
    _trainingTimer?.cancel();
    _quantumTimer?.cancel();
    super.dispose();
  }

  void _initializePQC() {
    Future.delayed(const Duration(seconds: 1), () {
      _addPQCLog('Generating Kyber-1024 keypair...', LogType.info);
    });

    Future.delayed(const Duration(seconds: 2), () {
      _addPQCLog(
        'Keypair generated: 1568 bytes public, 3168 bytes private',
        LogType.success,
      );
      _addPQCLog('Initiating key encapsulation with server...', LogType.info);
    });

    Future.delayed(const Duration(seconds: 3), () {
      _addPQCLog('Key encapsulation successful', LogType.success);
      _addPQCLog('Shared secret established: 32 bytes', LogType.success);
      _addPQCLog('Generating Dilithium-5 signature keypair...', LogType.info);
    });

    Future.delayed(const Duration(seconds: 4), () {
      _addPQCLog('Signature keypair ready: 2592 bytes public', LogType.success);
      _addPQCLog('PQC Tunnel Established ✓', LogType.success);
      setState(() => _isPQCConnected = true);
    });
  }

  void _initializeQuantumSimulation() {
    _quantumTimer = Timer.periodic(const Duration(milliseconds: 500), (timer) {
      if (_quantumStates.length >= 100) {
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

    _addPQCLog('Starting Federated Learning...', LogType.info);

    _trainingTimer = Timer.periodic(const Duration(seconds: 2), (timer) {
      if (_currentRound >= _totalRounds) {
        timer.cancel();
        setState(() => _isTraining = false);
        _addPQCLog(
          'Training complete! Final accuracy: ${(_currentAccuracy * 100).toStringAsFixed(1)}%',
          LogType.success,
        );
        return;
      }

      final random = Random();
      final newLoss =
          2.0 * exp(-0.3 * _currentRound) + random.nextDouble() * 0.1;
      final newAccuracy = min(
        0.98,
        0.5 + 0.05 * _currentRound + random.nextDouble() * 0.02,
      );

      setState(() {
        _currentRound++;
        _currentLoss = newLoss;
        _currentAccuracy = newAccuracy;
        _lossHistory.add(newLoss);
        _accuracyHistory.add(newAccuracy);
      });

      _addPQCLog(
        'Round $_currentRound: loss=${newLoss.toStringAsFixed(4)}, acc=${(newAccuracy * 100).toStringAsFixed(1)}%',
        LogType.info,
      );

      if (_currentRound % 3 == 0) {
        _addPQCLog(
          'Quantum aggregation triggered (VQC update)',
          LogType.quantum,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0A0E21),
      appBar: _buildAppBar(),
      body: Column(
        children: [
          _buildStatusBar(),
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
      floatingActionButton: _buildFAB(),
    );
  }

  PreferredSizeWidget _buildAppBar() {
    return AppBar(
      backgroundColor: Colors.transparent,
      elevation: 0,
      title: Row(
        children: [
          AnimatedBuilder(
            animation: _pulseController,
            builder: (context, child) {
              return Container(
                width: 36,
                height: 36,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  gradient: RadialGradient(
                    colors: [
                      const Color(
                        0xFF6C63FF,
                      ).withOpacity(0.3 + 0.3 * _pulseController.value),
                      Colors.transparent,
                    ],
                  ),
                ),
                child: const Center(
                  child: Text(
                    'Q',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: Color(0xFF6C63FF),
                    ),
                  ),
                ),
              );
            },
          ),
          const SizedBox(width: 8),
          const Text(
            'Q-Edge',
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.w600,
              letterSpacing: 1,
            ),
          ),
          const SizedBox(width: 8),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
            decoration: BoxDecoration(
              color: const Color(0xFF6C63FF).withOpacity(0.2),
              borderRadius: BorderRadius.circular(4),
            ),
            child: const Text(
              'FQNN',
              style: TextStyle(
                fontSize: 10,
                fontWeight: FontWeight.w500,
                color: Color(0xFF6C63FF),
              ),
            ),
          ),
        ],
      ),
      actions: [_buildConnectionIndicator(), const SizedBox(width: 16)],
    );
  }

  Widget _buildConnectionIndicator() {
    return Row(
      children: [
        Container(
          width: 8,
          height: 8,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: _isPQCConnected ? Colors.green : Colors.red,
            boxShadow: [
              BoxShadow(
                color: (_isPQCConnected ? Colors.green : Colors.red)
                    .withOpacity(0.5),
                blurRadius: 4,
                spreadRadius: 1,
              ),
            ],
          ),
        ),
        const SizedBox(width: 8),
        Text(
          _isPQCConnected ? 'PQC Secured' : 'Connecting...',
          style: TextStyle(
            fontSize: 12,
            color: _isPQCConnected ? Colors.green : Colors.orange,
          ),
        ),
      ],
    );
  }

  Widget _buildStatusBar() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1F3C),
        border: Border(
          bottom: BorderSide(color: Colors.white.withOpacity(0.1)),
        ),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _buildStatusItem(
            'Round',
            '$_currentRound/$_totalRounds',
            Icons.loop,
            Colors.blue,
          ),
          _buildStatusItem(
            'Loss',
            _currentLoss.toStringAsFixed(4),
            Icons.trending_down,
            Colors.orange,
          ),
          _buildStatusItem(
            'Accuracy',
            '${(_currentAccuracy * 100).toStringAsFixed(1)}%',
            Icons.check_circle_outline,
            Colors.green,
          ),
          _buildStatusItem(
            'Status',
            _isTraining ? 'Training' : 'Idle',
            _isTraining ? Icons.sync : Icons.pause,
            _isTraining ? Colors.purple : Colors.grey,
          ),
        ],
      ),
    );
  }

  Widget _buildStatusItem(
    String label,
    String value,
    IconData icon,
    Color color,
  ) {
    return Column(
      children: [
        Icon(icon, color: color, size: 20),
        const SizedBox(height: 4),
        Text(
          value,
          style: TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.w600,
            color: color,
          ),
        ),
        Text(label, style: TextStyle(fontSize: 10, color: Colors.grey[400])),
      ],
    );
  }

  Widget _buildTabBar() {
    return Container(
      color: const Color(0xFF1A1F3C),
      child: TabBar(
        controller: _tabController,
        indicatorColor: const Color(0xFF6C63FF),
        labelColor: const Color(0xFF6C63FF),
        unselectedLabelColor: Colors.grey,
        tabs: const [
          Tab(icon: Icon(Icons.insights), text: 'Training'),
          Tab(icon: Icon(Icons.blur_on), text: 'Quantum'),
          Tab(icon: Icon(Icons.security), text: 'Security'),
          Tab(icon: Icon(Icons.settings), text: 'Settings'),
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
          _buildSectionTitle('Loss Curve'),
          const SizedBox(height: 8),
          _buildLossChart(),
          const SizedBox(height: 24),
          _buildSectionTitle('Accuracy Curve'),
          const SizedBox(height: 8),
          _buildAccuracyChart(),
          const SizedBox(height: 24),
          _buildSectionTitle('Training Configuration'),
          const SizedBox(height: 8),
          _buildConfigCard(),
        ],
      ),
    );
  }

  Widget _buildLossChart() {
    return Container(
      height: 200,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1F3C),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withOpacity(0.1)),
      ),
      child: CustomPaint(
        size: const Size(double.infinity, 168),
        painter: ChartPainter(
          data: _lossHistory,
          color: Colors.orange,
          maxY: 2.5,
        ),
      ),
    );
  }

  Widget _buildAccuracyChart() {
    return Container(
      height: 200,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1F3C),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withOpacity(0.1)),
      ),
      child: CustomPaint(
        size: const Size(double.infinity, 168),
        painter: ChartPainter(
          data: _accuracyHistory,
          color: Colors.green,
          maxY: 1.0,
        ),
      ),
    );
  }

  Widget _buildConfigCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1F3C),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withOpacity(0.1)),
      ),
      child: Column(
        children: [
          _buildConfigRow('Local Epochs', '5'),
          _buildConfigRow('Batch Size', '32'),
          _buildConfigRow('Learning Rate', '0.001'),
          _buildConfigRow('Gradient Compression', '10%'),
          _buildConfigRow('Aggregation', 'FedAvg + VQC'),
        ],
      ),
    );
  }

  Widget _buildConfigRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: TextStyle(color: Colors.grey[400])),
          Text(value, style: const TextStyle(fontWeight: FontWeight.w600)),
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
          _buildSectionTitle('Quantum Entanglement Status'),
          const SizedBox(height: 8),
          _buildEntanglementVisualizer(),
          const SizedBox(height: 24),
          _buildSectionTitle('Qubit States'),
          const SizedBox(height: 8),
          _buildQubitStates(),
          const SizedBox(height: 24),
          _buildSectionTitle('VQC Parameters'),
          const SizedBox(height: 8),
          _buildVQCParams(),
        ],
      ),
    );
  }

  Widget _buildEntanglementVisualizer() {
    final latestState = _quantumStates.isNotEmpty
        ? _quantumStates.last
        : QuantumState();

    return Container(
      height: 200,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1F3C),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withOpacity(0.1)),
      ),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _buildQubitCircle('Q0', latestState.qubit0, Colors.blue),
              AnimatedBuilder(
                animation: _pulseController,
                builder: (context, child) {
                  return Container(
                    width: 100,
                    height: 4,
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        colors: [
                          Colors.blue.withOpacity(0.3),
                          Colors.purple.withOpacity(
                            0.8 * latestState.entanglement,
                          ),
                          Colors.red.withOpacity(0.3),
                        ],
                      ),
                      borderRadius: BorderRadius.circular(2),
                    ),
                  );
                },
              ),
              _buildQubitCircle('Q1', latestState.qubit1, Colors.red),
            ],
          ),
          const SizedBox(height: 24),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _buildMetricBox(
                'Entanglement',
                '${(latestState.entanglement * 100).toStringAsFixed(1)}%',
                Colors.purple,
              ),
              _buildMetricBox(
                'Coherence',
                '${(latestState.coherence * 100).toStringAsFixed(1)}%',
                Colors.cyan,
              ),
              _buildMetricBox(
                'Fidelity',
                '${(0.95 + 0.05 * latestState.coherence * 100).toStringAsFixed(1)}%',
                Colors.green,
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildQubitCircle(String label, double value, Color color) {
    return Column(
      children: [
        Container(
          width: 60,
          height: 60,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            gradient: RadialGradient(
              colors: [color.withOpacity(0.8), color.withOpacity(0.2)],
            ),
            boxShadow: [
              BoxShadow(
                color: color.withOpacity(0.4),
                blurRadius: 12,
                spreadRadius: 2,
              ),
            ],
          ),
          child: Center(
            child: Text(
              '|${value > 0.5 ? "1" : "0"}⟩',
              style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
          ),
        ),
        const SizedBox(height: 8),
        Text(label, style: TextStyle(color: Colors.grey[400])),
      ],
    );
  }

  Widget _buildMetricBox(String label, String value, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Column(
        children: [
          Text(
            value,
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.bold,
              color: color,
            ),
          ),
          Text(
            label,
            style: TextStyle(fontSize: 10, color: color.withOpacity(0.8)),
          ),
        ],
      ),
    );
  }

  Widget _buildQubitStates() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1F3C),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withOpacity(0.1)),
      ),
      child: Column(
        children: List.generate(8, (index) {
          final random = Random(index);
          final amplitude = 0.3 + random.nextDouble() * 0.7;

          return Padding(
            padding: const EdgeInsets.symmetric(vertical: 4),
            child: Row(
              children: [
                Text(
                  'q$index:',
                  style: TextStyle(
                    fontFamily: 'monospace',
                    color: Colors.grey[400],
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: LinearProgressIndicator(
                    value: amplitude,
                    backgroundColor: Colors.grey[800],
                    valueColor: AlwaysStoppedAnimation(
                      Color.lerp(Colors.blue, Colors.purple, index / 7)!,
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                Text(
                  amplitude.toStringAsFixed(2),
                  style: const TextStyle(fontFamily: 'monospace'),
                ),
              ],
            ),
          );
        }),
      ),
    );
  }

  Widget _buildVQCParams() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1F3C),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withOpacity(0.1)),
      ),
      child: Column(
        children: [
          _buildConfigRow('Qubits', '8'),
          _buildConfigRow('Layers', '4'),
          _buildConfigRow('Ansatz', 'StronglyEntangling'),
          _buildConfigRow('Entanglement', 'Full'),
          _buildConfigRow('Parameters', '96'),
          _buildConfigRow('Circuit Depth', '17'),
        ],
      ),
    );
  }

  Widget _buildSecurityTab() {
    return Column(
      children: [
        Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: const Color(0xFF1A1F3C),
            border: Border(
              bottom: BorderSide(color: Colors.white.withOpacity(0.1)),
            ),
          ),
          child: Row(
            children: [
              Icon(
                _isPQCConnected ? Icons.lock : Icons.lock_open,
                color: _isPQCConnected ? Colors.green : Colors.orange,
              ),
              const SizedBox(width: 8),
              Text(
                _isPQCConnected
                    ? 'Post-Quantum Cryptography Active'
                    : 'Establishing Secure Connection...',
                style: TextStyle(
                  fontWeight: FontWeight.w600,
                  color: _isPQCConnected ? Colors.green : Colors.orange,
                ),
              ),
            ],
          ),
        ),
        Expanded(
          child: ListView.builder(
            padding: const EdgeInsets.all(16),
            itemCount: _pqcLogs.length,
            itemBuilder: (context, index) {
              final log = _pqcLogs[_pqcLogs.length - 1 - index];
              return _buildLogEntry(log);
            },
          ),
        ),
      ],
    );
  }

  Widget _buildLogEntry(PQCLogEntry log) {
    final color = _getLogColor(log.type);
    final icon = _getLogIcon(log.type);

    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: color, size: 16),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(log.message, style: TextStyle(color: color)),
                const SizedBox(height: 4),
                Text(
                  _formatTime(log.timestamp),
                  style: TextStyle(fontSize: 10, color: Colors.grey[500]),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Color _getLogColor(LogType type) {
    switch (type) {
      case LogType.info:
        return Colors.blue;
      case LogType.success:
        return Colors.green;
      case LogType.warning:
        return Colors.orange;
      case LogType.error:
        return Colors.red;
      case LogType.quantum:
        return Colors.purple;
    }
  }

  IconData _getLogIcon(LogType type) {
    switch (type) {
      case LogType.info:
        return Icons.info_outline;
      case LogType.success:
        return Icons.check_circle_outline;
      case LogType.warning:
        return Icons.warning_amber;
      case LogType.error:
        return Icons.error_outline;
      case LogType.quantum:
        return Icons.blur_on;
    }
  }

  String _formatTime(DateTime time) {
    return '${time.hour.toString().padLeft(2, '0')}:'
        '${time.minute.toString().padLeft(2, '0')}:'
        '${time.second.toString().padLeft(2, '0')}.'
        '${time.millisecond.toString().padLeft(3, '0')}';
  }

  Widget _buildSettingsTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildSectionTitle('Server Configuration'),
          const SizedBox(height: 8),
          _buildSettingsCard([
            _buildSettingTile(
              'Server Address',
              'api.qedge.ai:443',
              Icons.cloud,
            ),
            _buildSettingTile(
              'Azure Quantum Workspace',
              'qedge-production',
              Icons.computer,
            ),
            _buildSettingTile('Quantum Target', 'ionq.simulator', Icons.memory),
          ]),
          const SizedBox(height: 24),
          _buildSectionTitle('Security Settings'),
          const SizedBox(height: 8),
          _buildSettingsCard([
            _buildSwitchTile(
              'Post-Quantum Cryptography',
              'Kyber-1024 + Dilithium-5',
              true,
            ),
            _buildSwitchTile(
              'Gradient Compression',
              'Top-K Sparsification (10%)',
              true,
            ),
            _buildSwitchTile(
              'Battery-Aware Training',
              'Adapt to device state',
              true,
            ),
          ]),
          const SizedBox(height: 24),
          _buildSectionTitle('About'),
          const SizedBox(height: 8),
          _buildSettingsCard([
            _buildSettingTile('Version', '1.0.0-beta', Icons.info),
            _buildSettingTile('Client ID', 'mobile_001', Icons.perm_identity),
            _buildSettingTile(
              'Research Lead',
              'Ahmad Rasidi (Roy)',
              Icons.person,
            ),
          ]),
        ],
      ),
    );
  }

  Widget _buildSettingsCard(List<Widget> children) {
    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF1A1F3C),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withOpacity(0.1)),
      ),
      child: Column(children: children),
    );
  }

  Widget _buildSettingTile(String title, String subtitle, IconData icon) {
    return ListTile(
      leading: Icon(icon, color: const Color(0xFF6C63FF)),
      title: Text(title),
      subtitle: Text(subtitle, style: TextStyle(color: Colors.grey[400])),
      trailing: const Icon(Icons.chevron_right),
    );
  }

  Widget _buildSwitchTile(String title, String subtitle, bool value) {
    return ListTile(
      title: Text(title),
      subtitle: Text(
        subtitle,
        style: TextStyle(color: Colors.grey[400], fontSize: 12),
      ),
      trailing: Switch(
        value: value,
        onChanged: (v) {},
        activeColor: const Color(0xFF6C63FF),
      ),
    );
  }

  Widget _buildSectionTitle(String title) {
    return Text(
      title,
      style: const TextStyle(
        fontSize: 16,
        fontWeight: FontWeight.w600,
        color: Color(0xFF6C63FF),
      ),
    );
  }

  Widget _buildFAB() {
    return FloatingActionButton.extended(
      onPressed: _isPQCConnected ? _startTraining : null,
      backgroundColor: _isPQCConnected
          ? (_isTraining ? Colors.red : const Color(0xFF6C63FF))
          : Colors.grey,
      icon: Icon(_isTraining ? Icons.stop : Icons.play_arrow),
      label: Text(_isTraining ? 'Stop' : 'Start Training'),
    );
  }
}


class ChartPainter extends CustomPainter {
  final List<double> data;
  final Color color;
  final double maxY;

  ChartPainter({required this.data, required this.color, required this.maxY});

  @override
  void paint(Canvas canvas, Size size) {
    if (data.isEmpty) {
      final textPainter = TextPainter(
        text: TextSpan(
          text: 'No data yet',
          style: TextStyle(color: Colors.grey[600]),
        ),
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();
      textPainter.paint(
        canvas,
        Offset(
          (size.width - textPainter.width) / 2,
          (size.height - textPainter.height) / 2,
        ),
      );
      return;
    }

    final paint = Paint()
      ..color = color
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;

    final fillPaint = Paint()
      ..shader = LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: [color.withOpacity(0.3), color.withOpacity(0.0)],
      ).createShader(Rect.fromLTWH(0, 0, size.width, size.height));

    final path = Path();
    final fillPath = Path();

    final dx = size.width / (data.length > 1 ? data.length - 1 : 1);

    for (var i = 0; i < data.length; i++) {
      final x = i * dx;
      final y = size.height - (data[i] / maxY) * size.height;

      if (i == 0) {
        path.moveTo(x, y);
        fillPath.moveTo(x, size.height);
        fillPath.lineTo(x, y);
      } else {
        path.lineTo(x, y);
        fillPath.lineTo(x, y);
      }
    }

    fillPath.lineTo((data.length - 1) * dx, size.height);
    fillPath.close();

    canvas.drawPath(fillPath, fillPaint);
    canvas.drawPath(path, paint);

    final dotPaint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;

    for (var i = 0; i < data.length; i++) {
      final x = i * dx;
      final y = size.height - (data[i] / maxY) * size.height;
      canvas.drawCircle(Offset(x, y), 3, dotPaint);
    }
  }

  @override
  bool shouldRepaint(covariant ChartPainter oldDelegate) {
    return oldDelegate.data != data;
  }
}


enum LogType { info, success, warning, error, quantum }

class PQCLogEntry {
  final String message;
  final LogType type;
  final DateTime timestamp;

  PQCLogEntry({
    required this.message,
    required this.type,
    required this.timestamp,
  });
}

class QuantumState {
  final double qubit0;
  final double qubit1;
  final double entanglement;
  final double coherence;
  final DateTime timestamp;

  QuantumState({
    this.qubit0 = 0.0,
    this.qubit1 = 0.0,
    this.entanglement = 0.5,
    this.coherence = 1.0,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();
}
