
import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:myoelectric_ftg/ble_log.dart';
import 'package:myoelectric_ftg/constant.dart';
import 'package:myoelectric_ftg/detail.dart';
import 'package:myoelectric_ftg/extra.dart';

class BleHandler extends StatefulWidget {
  const BleHandler({super.key});

  @override
  State<BleHandler> createState() => _BleHandlerState();
}

class _BleHandlerState extends State<BleHandler> {
  late StreamSubscription<List<ScanResult>> _scanResultsSubscription;
  BluetoothAdapterState _adapterState = BluetoothAdapterState.unknown;
  late StreamSubscription<BluetoothAdapterState> _adapterStateStateSubscription;
  final bleLogsBk = GlobalKey<BleLogState>();
  bool _isScanBtnDisabled = false;
  ScanResult? _bleDevice;
  Timer? _timer;
  List<String> logs = [];

  @override
  void initState() {
    super.initState();
    _adapterStateStateSubscription = FlutterBluePlus.adapterState.listen((
      state,
    ) {
      _adapterState = state;
    });
    _scanResultsSubscription = FlutterBluePlus.scanResults.listen(
      (results) {
        final bleDevices = results
            .where(
              (r) => r.advertisementData.advName.toUpperCase().startsWith('KT'),
            )
            .toList();

        if (bleDevices.isNotEmpty) {
          _timer?.cancel();
          final bleDevice = bleDevices.first;
          _bleDevice = bleDevice;
          setState(() {
            bleLogsBk.currentState?.addLog(
              '发现设备${bleDevice.advertisementData.advName}',
            );
          });

          if (!bleDevice.device.isConnected) {
            FlutterBluePlus.stopScan();
            bleDevice.device.connectAndUpdateStream().catchError((e) {
              setState(() {
                _isScanBtnDisabled = false;
                bleLogsBk.currentState?.addLog('蓝牙连接失败');
              });
            });
            setState(() {
              bleLogsBk.currentState?.addLog('创建蓝牙连接');
            });

            setState(() => _isScanBtnDisabled = false);
            Future.delayed(const Duration(milliseconds: 1500), () {
              final route = MaterialPageRoute(
                builder: (context) => Detail(device: bleDevice.device),
              );
              Navigator.of(context).push(route);
            });
          }
        }
      },
      onDone: () {
        if (_bleDevice == null) {
          setState(() {
            bleLogsBk.currentState?.addLog('未发现设备');
          });
        } else {
          _bleDevice = null;
        }
        setState(() => _isScanBtnDisabled = false);
      },
      onError: (e) {
        setState(() {
          bleLogsBk.currentState?.addLog('蓝牙连接失败');
        });
      },
    );
  }

  @override
  void dispose() {
    _scanResultsSubscription.cancel();
    _adapterStateStateSubscription.cancel();
    _timer?.cancel();
    super.dispose();
  }

  Future onScanDevices() async {
    setState(() => _isScanBtnDisabled = true);
    if (_adapterState != BluetoothAdapterState.on) {
      bleLogsBk.currentState?.addLog('蓝牙未打开或未授权');
      setState(() => _isScanBtnDisabled = false);
      return;
    }

    try {
      bleLogsBk.currentState?.addLog('开始搜索蓝牙');
      _timer = Timer(const Duration(milliseconds: 15500), () {
        if (mounted) {
          if (_bleDevice == null) {
            bleLogsBk.currentState?.addLog('未发现设备');
          } else {
            _bleDevice = null;
          }
          setState(() => _isScanBtnDisabled = false);
        }
      });
      await FlutterBluePlus.startScan(timeout: const Duration(seconds: 15));
    } catch (e) {
      _timer?.cancel();
      debugPrint('onScanDevices Error: $e');
      setState(() => _isScanBtnDisabled = false);
      bleLogsBk.currentState?.addLog('发生错误');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      resizeToAvoidBottomInset: false,
      body: Container(
        color: const Color(0xFF2C2F4B),
        padding: const EdgeInsets.symmetric(horizontal: 24),
        child: Column(
          children: [
            statusBarPlaceholder(),
            Expanded(child: Container()),
            Expanded(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20),
                child: BleLog(
                  key: bleLogsBk,
                  color: progressBkgColor.withOpacity(0.8),
                ),
              ),
            ),
            const SizedBox(height: 30),
            FractionallySizedBox(
              widthFactor: 0.8,
              child: ElevatedButton(
                onPressed: _isScanBtnDisabled ? null : onScanDevices,
                child: const Text('扫描设备'),
              ),
            ),
            const SizedBox(height: 55),
          ],
        ),
      ),
    );
  }

  Widget statusBarPlaceholder() {
    final statusBarHeight = MediaQuery.of(context).padding.top;
    return Container(height: statusBarHeight);
  }
}