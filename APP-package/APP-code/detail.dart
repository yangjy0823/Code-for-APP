import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:myoelectric_ftg/api.dart';
import 'package:myoelectric_ftg/arrow_quene.dart';
import 'package:myoelectric_ftg/bean/base.dart';
import 'package:myoelectric_ftg/bean/m_ftg.dart';
import 'package:myoelectric_ftg/ble_log.dart';
import 'package:myoelectric_ftg/constant.dart';
import 'package:myoelectric_ftg/extra.dart';
import 'package:myoelectric_ftg/fatigue_level.dart';
import 'package:myoelectric_ftg/lines_chart.dart';
import 'package:toastification/toastification.dart';
import 'package:uuid/uuid.dart';

class Detail extends StatefulWidget {
  final BluetoothDevice device;

  const Detail({super.key, required this.device});

  @override
  State<StatefulWidget> createState() => _DetailState();
}

class _DetailState extends State<Detail> with SingleTickerProviderStateMixin {
  BluetoothConnectionState _connectionState =
      BluetoothConnectionState.disconnected;
  List<BluetoothService> _services = [];
  late StreamSubscription<BluetoothConnectionState>
  _connectionStateSubscription;
  late BluetoothCharacteristic _characteristicWrite;
  StreamSubscription<List<int>>? _lastValueSubscription;
  final bleLogGk = GlobalKey<BleLogState>();
  String info = '';
  String bluetoothMacAddr = '';

  late TabController _tabController;
  List tabs = ["Real-time sensor data", "Fatigue Analysis"];

  // 钠离子浓度 （ mmol/L ）
  double cNa = 0.0;
  // 钾离子浓度 (mmol/L)
  double cK = 0.0;
  // 尿素浓度 (mmol/L)
  double cUrea = 0.0;
  // 葡萄糖浓度 (mmol/L)
  double cGlucose = 0.0;
  // 乳酸浓度 (mmol/L)
  double cLactate = 0.0;
  // NH4浓度 (mmol/L)
  double cNH4 = 0.0;

  // 皮质醇浓度 (μg/L)
  double cCortisol = 0.0;
  // 睾酮浓度 (ng/L)
  double cTestosterone = 0.0;

  // 温度（℃）
  double temperature = 0.0;

  List<String> legends = ["Urea(mM)", "Lac(mM)", "K(mM)", "NH4(mM)"];
  List<String> legends2 = ["Cor(μg/L)", "Tes(ng/L)", "Na(mM)", "Glu(mM)"];
  List<String> legends3 = ["MEF", "MDF"];
  List<List<double?>> yAxisDataArr = List.generate(
    4,
    (index) => List.generate(300, (i) => null),
  );
  List<List<double?>> yAxisDataArr2 = List.generate(
    4,
    (index) => List.generate(300, (i) => null),
  );
  List<List<double?>> yAxisDataArr3 = List.generate(
    2,
    (index) => List.generate(60, (i) => null),
  );
  List<int?> tsList = List.generate(300, (index) => null);
  List<int?> tsList2 = List.generate(60, (index) => null);

  int startTs = 0;
  bool startReceivedMsg = false;
  String xAxisName = "0.00 min";
  String xAxisName2 = "0.00 min";

  Map<String, dynamic> param1 = {"uuid": null, "id": null, "emg": []};
  Map<String, int?> param1Data = {"sEMG": null, "Time": null};
  Map<String, dynamic> postParam = {"uuid": null, "id": null, "data": []};
  Map<String, double?> postParamData = {
    // 钠离子浓度 （ mmol/L ）
    "cNa": null,
    // 钾离子浓度 (mmol/L)
    "cK": null,
    // 尿素浓度 (mmol/L)
    "cUrea": null,
    // 葡萄糖浓度 (mmol/L)
    "cGlucose": null,
    // 乳酸浓度 (mmol/L)
    "cLactate": null,
    // NH4浓度 (mmol/L)
    "cNH4": null,
    // 皮质醇浓度 (μg/L)
    "cCortisol": null,
    // 睾酮浓度 (ng/L)
    "cTestosterone": null,
    // 温度（℃）
    "temperature": null,
  };

  List<String> ids = [];

  bool showLoading = false;

  var uuid = Uuid();

  List<List<int>> modelOutputLabels = List.generate(
    6,
    (index) => List.generate(6, (i) => -1),
  );
  List<String> arrowQueneXAxis = List.generate(12, (index) => "");
  int arrowChartStartTs = 0;
  String arrowChartXAxisName = '0 min';
  int arrowQueneRecvCnt = 0;
  int arrowQueneRecvLastTs = 0;
  int arrowQueneLimitDownTs = 0;

  var supplyAdvise = "";

  bool isFinishAnalysis = false;

  double fLevel = 0.0;

  @override
  void initState() {
    super.initState();

    _tabController = TabController(length: tabs.length, vsync: this);
    setState(() {
      supplyAdvise = calcAdvise(modelOutputLabels);
    });

    SystemChrome.setPreferredOrientations([
      DeviceOrientation.portraitUp,
      DeviceOrientation.landscapeLeft,
      DeviceOrientation.landscapeRight,
    ]);

    _connectionStateSubscription = widget.device.connectionState.listen((
      state,
    ) async {
      if (state == BluetoothConnectionState.connected) {
        _services = [];
        onDiscoverServices();
      } else if (state == BluetoothConnectionState.disconnected) {
        debugPrint('蓝牙已断开连接');
      }
      setState(() {
        _connectionState = state;
      });
    });
  }

  Future onDiscoverServices() async {
    try {
      _services = await widget.device.discoverServices();
      if (_services.isEmpty) {
        throw Exception('获取蓝牙服务失败');
      }

      for (final s in _services) {
        for (final c in s.characteristics) {
          if (c.properties.notify) {
            setState(() {
              info = 'Bluetooth Service ${s.remoteId.str}${s.serviceUuid.str}';
              bluetoothMacAddr = '${s.remoteId.str}${s.serviceUuid.str}';
            });
            _characteristicWrite = c;
            if (!isFinishAnalysis) {
              setReceivedMsg3();
            }
            await c.setNotifyValue(true);
          }
        }
      }
    } catch (e) {
      debugPrint('获取蓝牙服务失败: $e');
      if (mounted) Navigator.pop(context);
    }
  }

  List<Map<String, int?>> recvEmgData = [];

  List<int> recvBiochemicalData = [];
  List<int> biochemicalData = [];

  int emgStartTs = 0;

  bool isStartHandleRecvData = false;
  bool isRecvBiochemicalData = false;

  int recvCnt = 0;
  int recvThresholdCnt = 9;

  final pageSize = 10;
  int get pageNo => (recvCnt / pageSize).ceil();
  int nowTs = 0;

  void setReceivedMsg3() {
    _lastValueSubscription = _characteristicWrite.lastValueStream.listen((
      codeUnits,
    ) {
      try {
        if (codeUnits.isNotEmpty && codeUnits.length % 2 == 0) {
          for (int i = 0; i < codeUnits.length; i += 2) {
            int value = (codeUnits[i + 1] << 8) | codeUnits[i];

            if (value == 6000) {
              isStartHandleRecvData = true;
              isRecvBiochemicalData = true;
            } else if (value == 7000) {
              isRecvBiochemicalData = false;
              biochemicalData = List.from(recvBiochemicalData);
              recvBiochemicalData.clear();
              _processBiochemicalData(biochemicalData);
            } else if (isStartHandleRecvData) {
              if (isRecvBiochemicalData) {
                recvBiochemicalData.add(value);
              } else {
                param1Data["sEMG"] = value;
                param1Data["Time"] = DateTime.now().millisecondsSinceEpoch;
                recvEmgData.add(param1Data);
                resetParam1Data();
              }
            }
          }
        }

        if (isStartHandleRecvData) {
          if (emgStartTs == 0) {
            emgStartTs = DateTime.now().millisecondsSinceEpoch;
          } else {
            Future<bool> uploadResult;
            List<Future<dynamic>> uploadArr = [];
            nowTs = DateTime.now().millisecondsSinceEpoch;
            if (nowTs - emgStartTs > 60_000 && isRecvBiochemicalData == false) {
              recvCnt++;
              emgStartTs = 0;
              String id = uuid.v8();
              ids.add(id);
              param1["uuid"] = bluetoothMacAddr;
              param1["id"] = id;
              param1["emg"] = recvEmgData;
              uploadResult = uploadEmgData(param1);
              uploadArr.add(uploadResult);
              resetParam1();
              if (recvCnt > recvThresholdCnt) {
                postParam["uuid"] = bluetoothMacAddr;
                postParam["id"] = ids[ids.length - recvThresholdCnt - 1];
                uploadResult = uploadBiochemialData(postParam);
                uploadArr.add(uploadResult);
              }
              resetPostParam();

              recvEmgData = [];
              Future.wait(uploadArr)
                  .then((value) async {
                    getMefAndMdfData();

                    var success = value.every((element) => element);
                    if (success) {
                      if (arrowQueneLimitDownTs == 0) {
                        arrowQueneLimitDownTs =
                            (DateTime.now().millisecondsSinceEpoch / 1000)
                                .toInt();
                      }
                      getMftgData();
                    }
                  })
                  .catchError((error) {
                    debugPrint("upload error: $error");
                  });
            }
          }
        }
      } catch (e) {
        debugPrint('$e');
      }
    });
  }

  var cnt = 0;
  void _processBiochemicalData(List<int> values) {
    // 根据协议处理特殊数据
    // v0: 电压K
    // v1: 电压UN
    // v2: 电流IL-6
    // v3: 电流Glu
    // v4: 电压N
    // v5: 电压Na
    // v6: 阻抗
    // v7: 电流Lac
    // v8: 电流Cor

    int voltageK = values[0];
    int voltageUN = values[1];
    int currentIL6 = values[2];
    int currentGlu = values[3];
    int voltageN = values[4];
    int voltageNa = values[5];
    int impedance = values[6];
    int currentLac = values[7];
    int currentCor = values[8];

    // 计算钾离子浓度
    if (voltageK <= 176) {
      cK = 0.0;
    } else {
      if (voltageK > 4096) {
        voltageK = 4096;
      }
      cK = (voltageK - 176) / 391.9;
    }

    // 计算尿素浓度
    if (voltageUN <= 395) {
      cUrea = 0.0;
    } else {
      if (voltageUN > 4096) {
        voltageUN = 4096;
      }
      cUrea = (voltageUN - 395) / 285;
    }

    // 计算睾酮浓度
    if (currentIL6 <= 512) {
      cTestosterone = 0.0;
    } else {
      if (currentIL6 > 4096) {
        currentIL6 = 4096;
      }
      cTestosterone = (currentIL6 - 512) / 37.5;
    }

    // 计算葡萄糖浓度
    if (currentGlu <= 200) {
      cGlucose = 0.0;
    } else {
      if (currentGlu > 4096) {
        currentGlu = 4096;
      }
      cGlucose = (currentGlu - 200) / 39.5;
    }

    // 计算NH4浓度
    if (voltageN <= 308) {
      cNH4 = 0.0;
    } else {
      if (voltageN > 4096) {
        voltageN = 4096;
      }
      cNH4 = (voltageN - 308) / 545;
    }

    // 计算钠离子浓度
    if (voltageNa <= 126) {
      cNa = 0.0;
    } else {
      if (voltageNa > 4096) {
        voltageNa = 4096;
      }
      cNa = (voltageNa - 126) / 48.4;
    }

    // 计算温度
    if (impedance <= 1005) {
      temperature = 0.0;
    } else {
      if (impedance > 4096) {
        impedance = 4096;
      }
      temperature = (impedance - 1005) / 45;
    }

    // 计算乳酸浓度
    if (currentLac <= 395) {
      cLactate = 0.0;
    } else {
      if (currentLac > 4096) {
        currentLac = 4096;
      }
      cLactate = (currentLac - 395) / 215;
    }

    // 计算皮质醇浓度
    if (currentCor <= 395) {
      cCortisol = 0.0;
    } else {
      if (currentCor > 4096) {
        currentCor = 4096;
      }
      cCortisol = (currentCor - 395) / 215;
    }

    postParamData["cK"] = cK;
    postParamData["cUrea"] = cUrea;
    postParamData["cTestosterone"] = cTestosterone;
    postParamData["cGlucose"] = cGlucose;
    postParamData["cNH4"] = cNH4;
    postParamData["cNa"] = cNa;
    postParamData["temperature"] = temperature;
    postParamData["cLactate"] = cLactate;
    postParamData["cCortisol"] = cCortisol;
    postParam["data"].add(postParamData);
    resetPostParamData();

    yAxisDataArr[0].removeAt(0);
    yAxisDataArr[0].add(cUrea);
    yAxisDataArr[1].removeAt(0);
    yAxisDataArr[1].add(cLactate);
    yAxisDataArr[2].removeAt(0);
    yAxisDataArr[2].add(cK);
    yAxisDataArr[3].removeAt(0);
    yAxisDataArr[3].add(cNH4);

    yAxisDataArr2[0].removeAt(0);
    yAxisDataArr2[0].add(cCortisol);
    yAxisDataArr2[1].removeAt(0);
    yAxisDataArr2[1].add(cTestosterone);
    yAxisDataArr2[2].removeAt(0);
    yAxisDataArr2[2].add(cNa);
    yAxisDataArr2[3].removeAt(0);
    yAxisDataArr2[3].add(cGlucose);

    int ts = DateTime.now().millisecondsSinceEpoch;
    double minutes = 0.0;
    if (tsList.last == null) {
      startTs = ts;
    } else {
      minutes = (DateTime.now().millisecondsSinceEpoch - startTs) / 60000;
    }
    tsList.removeAt(0);
    tsList.add(ts);
    setState(() {
      xAxisName = "${minutes.toStringAsFixed(2)} min";
    });
  }

  Future<bool> uploadEmgData(Map<String, dynamic> param) async {
    var res = await queryEmgAdd(param1);
    var success = false;
    if (200 <= res.statusCode && res.statusCode < 300) {
      final jsonObj = jsonDecode(res.body);
      final result = Result.fromJSON(jsonObj);
      if (200 <= result.code && result.code < 300) {
        success = true;
      }
    }
    if (!success) {
      debugPrint("O_O uploadEmgData Failed");
      toastification.show(
        context: context,
        type: ToastificationType.error,
        style: ToastificationStyle.flat,
        title: const Text('EMG数据上传失败'),
        alignment: Alignment.center,
        autoCloseDuration: const Duration(seconds: 1),
        borderRadius: BorderRadius.circular(12.0),
        boxShadow: highModeShadow,
        showProgressBar: false,
        closeButtonShowType: CloseButtonShowType.none,
        closeOnClick: false,
      );
    } else {
      debugPrint("O_O uploadEmgData Success");
    }
    return success;
  }

  Future<bool> uploadBiochemialData(Map<String, dynamic> param) async {
    var res = await queryBiochemialAdd(param);
    var success = false;
    if (200 <= res.statusCode && res.statusCode < 300) {
      final jsonObj = jsonDecode(res.body);
      final result = Result.fromJSON(jsonObj);
      if (200 <= result.code && result.code < 300) {
        success = true;
      }
    }
    if (!success) {
      toastification.show(
        context: context,
        type: ToastificationType.error,
        style: ToastificationStyle.flat,
        title: const Text('生化数据上传失败'),
        alignment: Alignment.center,
        autoCloseDuration: const Duration(seconds: 1),
        borderRadius: BorderRadius.circular(12.0),
        boxShadow: highModeShadow,
        showProgressBar: false,
        closeButtonShowType: CloseButtonShowType.none,
        closeOnClick: false,
      );
    } else {
      debugPrint("O_O uploadBiochemialData Success");
    }
    return success;
  }

  Future<bool> queryExecuteHandleData() async {
    var res = await queryExecute(null);
    var success = false;
    if (200 <= res.statusCode && res.statusCode < 300) {
      final jsonObj = jsonDecode(res.body);
      final result = Result.fromJSON(jsonObj);
      if (200 <= result.code && result.code < 300) {
        success = true;
      }
    }

    if (!success) {
      toastification.show(
        context: context,
        type: ToastificationType.error,
        style: ToastificationStyle.flat,
        title: const Text('执行后端数据处理失败'),
        alignment: Alignment.center,
        autoCloseDuration: const Duration(seconds: 1),
        borderRadius: BorderRadius.circular(12.0),
        boxShadow: highModeShadow,
        showProgressBar: false,
        closeButtonShowType: CloseButtonShowType.none,
        closeOnClick: false,
      );
    }
    return success;
  }

  var startMefMdfTs = 0;
  Future<bool> getMefAndMdfData() async {
    var res = await getEmgList(pageNo: pageNo, pageSize: pageSize);
    var success = false;
    if (200 <= res.statusCode && res.statusCode < 300) {
      final jsonObj = jsonDecode(res.body);
      final result = MdfMefResult.fromJSON(jsonObj);
      if (200 <= result.code && result.code < 300) {
        success = true;
        if (result.data != null && result.data!.isNotEmpty) {
          var lastData = result.data!.last;
          setState(() {
            yAxisDataArr3[0].removeAt(0);
            yAxisDataArr3[0].add(lastData.mef);
            yAxisDataArr3[1].removeAt(0);
            yAxisDataArr3[1].add(lastData.mdf);
            double minutes = 0.0;
            if (tsList2.last == null) {
              startMefMdfTs = nowTs;
              minutes = 1.0;
            } else {
              minutes = (nowTs - startMefMdfTs) / 60000 + 1;
            }
            xAxisName2 = "${minutes.toStringAsFixed(2)} min";
            tsList2.removeAt(0);
            tsList2.add(nowTs);
          });
        }
      }
    }

    if (!success) {
      toastification.show(
        context: context,
        type: ToastificationType.error,
        style: ToastificationStyle.flat,
        title: const Text('获取MEF和MDF数据失败'),
        alignment: Alignment.center,
        autoCloseDuration: const Duration(seconds: 1),
        borderRadius: BorderRadius.circular(12.0),
        boxShadow: highModeShadow,
        showProgressBar: false,
        closeButtonShowType: CloseButtonShowType.none,
        closeOnClick: false,
      );
    }
    return success;
  }

  Future<bool> getMftgData() async {
    var res = await getBiochemialList(pageNo: 1, pageSize: 12);
    var success = false;
    if (200 <= res.statusCode && res.statusCode < 300) {
      final jsonObj = jsonDecode(res.body);
      final result = MftgResult.fromJSON(jsonObj);
      if (200 <= result.code && result.code < 300) {
        success = true;
        if (result.data != null && result.data!.isNotEmpty) {
          var lastData = result.data!.last;
          if (lastData.xTime < arrowQueneLimitDownTs) {
            return success;
          }
          if (arrowChartStartTs == 0) {
            arrowChartStartTs = lastData.xTime;
            arrowQueneRecvLastTs = lastData.xTime;
            arrowChartXAxisName = "15 min";
          } else {
            final minutes = (lastData.xTime - arrowChartStartTs) / 60;
            arrowChartXAxisName = "${(minutes + 15).toStringAsFixed(0)} min";
            if (lastData.xTime == arrowQueneRecvLastTs) {
              return success;
            } else {
              arrowQueneRecvLastTs = lastData.xTime;
            }
          }

          int index = modelOutputLabels[0].indexOf(-1);
          if (index != -1) {
            modelOutputLabels[0][index] = lastData.glucoseClass;
            modelOutputLabels[1][index] = lastData.lactateClass;
            modelOutputLabels[2][index] = lastData.hydrationClass;
            modelOutputLabels[3][index] = lastData.proteinSupplyClass;
            modelOutputLabels[4][index] = lastData.muscleFatigueClass;
            modelOutputLabels[5][index] = lastData.fatigueClass;
          } else {
            modelOutputLabels[0].removeAt(0);
            modelOutputLabels[0].add(lastData.glucoseClass);
            modelOutputLabels[1].removeAt(0);
            modelOutputLabels[1].add(lastData.lactateClass);
            modelOutputLabels[2].removeAt(0);
            modelOutputLabels[2].add(lastData.hydrationClass);
            modelOutputLabels[3].removeAt(0);
            modelOutputLabels[3].add(lastData.proteinSupplyClass);
            modelOutputLabels[4].removeAt(0);
            modelOutputLabels[4].add(lastData.muscleFatigueClass);
            modelOutputLabels[5].removeAt(0);
            modelOutputLabels[5].add(lastData.fatigueClass);
          }

          setState(() {
            fLevel =
                (lastData.glucoseClass +
                    lastData.lactateClass +
                    lastData.hydrationClass +
                    lastData.proteinSupplyClass +
                    lastData.muscleFatigueClass +
                    lastData.fatigueClass) /
                6.0;
            supplyAdvise = calcAdvise(modelOutputLabels);
            debugPrint("O_O ---supplyAdvise: $supplyAdvise");
          });
        }
      }
    }
    return success;
  }

  Future onConnect() async {
    try {
      await widget.device.connectAndUpdateStream();
    } catch (e) {
      if (e is FlutterBluePlusException &&
          e.code == FbpErrorCode.connectionCanceled.index) {
      } else {}
    }
  }

  Future onDisconnect() async {
    try {
      await widget.device.disconnectAndUpdateStream();
    } catch (e) {
      debugPrint('onDisconnectPressed Error: $e');
    }
  }

  @override
  void dispose() {
    _connectionStateSubscription.cancel();
    _lastValueSubscription?.cancel();
    _tabController.dispose();
    super.dispose();
  }

  bool get isConnected {
    return _connectionState == BluetoothConnectionState.connected;
  }

  @override
  Widget build(BuildContext context) {
    final screenWidth = MediaQuery.of(context).size.width;
    return Scaffold(
      appBar: PreferredSize(
        preferredSize: const Size.fromHeight(
          kTextTabBarHeight,
        ), 
        child: SafeArea(
          bottom: false,
          child: Material(
            color: pageBkgColor,
            child: TabBar(
              controller: _tabController,
              dividerColor: Colors.black26,
              indicatorColor: progressBkgColor,
              labelColor: progressBkgColor,
              unselectedLabelColor: progressBkgColor.withAlpha(100),
              tabs: tabs.map((e) => Tab(text: e)).toList(),
            ),
          ),
        ),
      ),
      backgroundColor: pageBkgColor,
      body: Stack(
        fit: StackFit.expand,
        children: [
          TabBarView(
            controller: _tabController,
            children: [
              SingleChildScrollView(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    const SizedBox(height: 18),
                    const SizedBox(height: 12),
                    Container(
                      height: 235,
                      padding: const EdgeInsets.symmetric(horizontal: 15),
                      child: LinesChart(
                        yAxisDataArr: yAxisDataArr,
                        maxY: 20,
                        xAxisName: xAxisName,
                        legends: legends,
                        tsList: tsList,
                        splitLines: [0, 50, 100],
                        splitLinesLabels: ["0", "10", "20"],
                      ),
                    ),
                    const SizedBox(height: 12),
                    Container(
                      height: 235,
                      padding: const EdgeInsets.symmetric(horizontal: 15),
                      child: LinesChart(
                        yAxisDataArr: yAxisDataArr2,
                        maxY: 250,
                        xAxisName: xAxisName,
                        legends: legends2,
                        tsList: tsList,
                        splitLines: [0, 50, 100],
                        splitLinesLabels: ["0", "125", "250"],
                      ),
                    ),
                    const SizedBox(height: 12),
                    Container(
                      height: 235,
                      padding: const EdgeInsets.symmetric(horizontal: 15),
                      child: LinesChart(
                        yAxisDataArr: yAxisDataArr3,
                        maxY: 150,
                        xAxisName: xAxisName2,
                        legends: legends3,
                        tsList: tsList2,
                        splitLines: [0, 50, 100],
                        splitLinesLabels: ["0", "75", "150"],
                      ),
                    ),
                    const SizedBox(height: 12),
                    Container(
                      height: 370,
                      padding: const EdgeInsets.fromLTRB(24, 0, 60, 0),
                      child: GridView.count(
                        crossAxisCount: 2,
                        childAspectRatio: 6,
                        children: [
                          const DefaultTextStyle(
                            style: TextStyle(
                              fontSize: 16,
                              color: progressBkgColor,
                            ),
                            child: Text.rich(
                              TextSpan(
                                children: [
                                  TextSpan(text: "Na"),
                                  WidgetSpan(
                                    alignment: PlaceholderAlignment.bottom,
                                    child: Text("+"),
                                    style: TextStyle(fontSize: 12),
                                  ),
                                  TextSpan(text: "："),
                                ],
                              ),
                            ),
                          ),
                          Text(
                            "$cNaStr  mmol/L",
                            style: const TextStyle(
                              fontSize: 16,
                              color: progressBkgColor,
                            ),
                          ),
                          const DefaultTextStyle(
                            style: TextStyle(
                              fontSize: 16,
                              color: progressBkgColor,
                            ),
                            child: Text.rich(
                              TextSpan(
                                children: [
                                  TextSpan(text: "K"),
                                  WidgetSpan(
                                    alignment: PlaceholderAlignment.bottom,
                                    child: Text("+"),
                                    style: TextStyle(fontSize: 12),
                                  ),
                                  TextSpan(text: "："),
                                ],
                              ),
                            ),
                          ),
                          Text(
                            "$cKStr  mmol/L",
                            style: const TextStyle(
                              fontSize: 16,
                              color: progressBkgColor,
                            ),
                          ),
                          const Text(
                            "Urea：",
                            style: TextStyle(
                              fontSize: 16,
                              color: progressBkgColor,
                            ),
                          ),
                          Text(
                            "$cUreaStr  mmol/L",
                            style: const TextStyle(
                              fontSize: 16,
                              color: progressBkgColor,
                            ),
                          ),
                          const Text(
                            "Glu：",
                            style: TextStyle(
                              fontSize: 16,
                              color: progressBkgColor,
                            ),
                          ),
                          Text(
                            "$cGlucoseStr  mmol/L",
                            style: const TextStyle(
                              fontSize: 16,
                              color: progressBkgColor,
                            ),
                          ),
                          const DefaultTextStyle(
                            style: TextStyle(
                              fontSize: 16,
                              color: progressBkgColor,
                            ),
                            child: Text.rich(
                              TextSpan(
                                children: [
                                  TextSpan(text: "NH4"),
                                  WidgetSpan(
                                    alignment: PlaceholderAlignment.bottom,
                                    child: Text("+"),
                                    style: TextStyle(fontSize: 12),
                                  ),
                                  TextSpan(text: "："),
                                ],
                              ),
                            ),
                          ),
                          Text(
                            "$cNH4Str  mmol/L",
                            style: const TextStyle(
                              fontSize: 16,
                              color: progressBkgColor,
                            ),
                          ),
                          const Text(
                            "Lac：",
                            style: TextStyle(
                              fontSize: 16,
                              color: progressBkgColor,
                            ),
                          ),
                          Text(
                            "$cLactateStr mmol/L",
                            style: const TextStyle(
                              fontSize: 16,
                              color: progressBkgColor,
                            ),
                          ),
                          const Text(
                            "Tes：",
                            style: TextStyle(
                              fontSize: 16,
                              color: progressBkgColor,
                            ),
                          ),
                          Text(
                            "$cTestosteroneStr ng/L",
                            style: const TextStyle(
                              fontSize: 16,
                              color: progressBkgColor,
                            ),
                          ),
                          const Text(
                            "Cor:",
                            style: TextStyle(
                              fontSize: 16,
                              color: progressBkgColor,
                            ),
                          ),
                          Text(
                            "$cCortisolStr  μg/L",
                            style: const TextStyle(
                              fontSize: 16,
                              color: progressBkgColor,
                            ),
                          ),
                          const Text(
                            "Temperature:",
                            style: TextStyle(
                              fontSize: 16,
                              color: progressBkgColor,
                            ),
                          ),
                          Text(
                            "$temperatureStr  ℃",
                            style: const TextStyle(
                              fontSize: 16,
                              color: progressBkgColor,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
              SingleChildScrollView(
                physics: AlwaysScrollableScrollPhysics(),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    const SizedBox(height: 18),
                    FatigueLevel(value: fLevel),
                    ArrowChart(
                      data: modelOutputLabels,
                      yAxisLabels: [
                        "Glucose\nsupply",
                        "Lactate\naccumulation",
                        "Hydration\nstatus",
                        "Protein\nsupply",
                        "Muscle\nfatigue",
                        "Subjective\nFatigue",
                      ],
                      xAxisLabels: arrowQueneXAxis,
                      width: screenWidth,
                      xAxisName: arrowChartXAxisName,
                      xAxisMode: 1,
                      labelColor: progressBkgColor,
                      xAxisNameWidth: 40,
                    ),
                    SizedBox(height: 15),
                    Padding(
                      padding: EdgeInsetsGeometry.symmetric(horizontal: 15),
                      child: Text(
                        supplyAdvise,
                        style: const TextStyle(
                          fontSize: 16,
                          color: progressBkgColor,
                        ),
                      ),
                    ),
                    const SizedBox(height: 300),
                  ],
                ),
              ),
            ],
          ),

          Positioned(
            bottom: 0,
            child: Container(
              color: pageBkgColor,
              padding: EdgeInsets.fromLTRB(0, 0, 0, 10),
              width: MediaQuery.of(context).size.width,
              child: Column(
                children: [
                  Container(height: 1, color: Colors.black26),
                  SizedBox(height: 5),
                  ElevatedButton(
                    onPressed: () {
                      onDisconnect();
                      setState(() {
                        isFinishAnalysis = true;
                      });
                    },
                    child: const Text('Finish exercising'),
                  ),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(
                        connectIconData,
                        color: progressBkgColor.withAlpha(127),
                        size: 16,
                      ),
                      Text(
                        connectStateDesc,
                        style: TextStyle(
                          fontSize: 16,
                          color: progressBkgColor.withAlpha(127),
                        ),
                      ),
                    ],
                  ),
                  Text(
                    info,
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 15,
                      color: progressBkgColor.withAlpha(127),
                    ),
                  ),
                ],
              ),
            ),
          ),
          Positioned(
            child: showLoading
                ? const Center(
                    child: SizedBox(
                      width: 70,
                      height: 70,
                      child: CircularProgressIndicator(
                        valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                        strokeWidth: 6.0,
                      ),
                    ),
                  )
                : const SizedBox(),
          ),
        ],
      ),
    );
  }

  Widget statusBarPlaceholder() {
    final statusBarHeight = MediaQuery.of(context).padding.top;
    return Container(height: statusBarHeight);
  }

  String get cUreaStr {
    if (cUrea <= 0) {
      return " - ";
    }
    return cUrea.toStringAsFixed(4);
  }

  String get cGlucoseStr {
    if (cGlucose <= 0) {
      return " - ";
    }
    return cGlucose.toStringAsFixed(4);
  }

  String get cTestosteroneStr {
    if (cTestosterone <= 0) {
      return " - ";
    }
    return cTestosterone.toStringAsFixed(4);
  }

  String get cNH4Str {
    if (cNH4 <= 0) {
      return " - ";
    }
    return cNH4.toStringAsFixed(4);
  }

  String get cNaStr {
    if (cNa <= 0) {
      return " - ";
    }
    return cNa.toStringAsFixed(4);
  }

  String get cLactateStr {
    if (cLactate <= 0) {
      return " - ";
    }
    return cLactate.toStringAsFixed(4);
  }

  String get cCortisolStr {
    if (cCortisol <= 0) {
      return " - ";
    }
    return cCortisol.toStringAsFixed(4);
  }

  String get temperatureStr {
    if (temperature <= 0) {
      return " - ";
    }
    return temperature.toStringAsFixed(1);
  }

  String get cKStr {
    if (cK <= 0) {
      return " - ";
    }
    return cK.toStringAsFixed(4);
  }

  IconData get connectIconData {
    if (isConnected) {
      return Icons.bluetooth_connected_rounded;
    }
    return Icons.bluetooth_disabled_rounded;
  }

  String get connectStateDesc {
    if (isConnected) {
      return "Device connected";
    }
    return "Device disconnected";
  }

  resetParam1() {
    param1 = {"id": null, "emg": []};
  }

  resetParam1Data() {
    param1Data = {"sEMG": null, "Time": null};
  }

  resetPostParam() {
    postParam = {"uuid": null, "id": null, "data": []};
  }

  resetPostParamData() {
    postParamData = {
      // 钠离子浓度 （ mmol/L ）
      "cNa": null,
      // 钾离子浓度 (mmol/L)
      "cK": null,
      // 尿素浓度 (mmol/L)
      "cUrea": null,
      // 葡萄糖浓度 (mmol/L)
      "cGlucose": null,
      // 乳酸浓度 (mmol/L)
      "cLactate": null,
      // NH4浓度 (mmol/L)
      "cNH4": null,
      // 皮质醇浓度 (μg/L)
      "cCortisol": null,
      // 睾酮浓度 (ng/L)
      "cTestosterone": null,
      // 温度（℃）
      "temperature": null,
    };
  }

  String calcAdvise(List<List<int>> modelOutputLabels) {
    var advise = "* Supply advise: ";
    List<String> supplyArr = [];

    int len = modelOutputLabels[0].length;
    int cols = len;
    for (int i = len - 1; i >= 0; i--) {
      if (modelOutputLabels[0][i] != -1) {
        cols = i + 1;
        break;
      }
    }
    final glucoseSupply = modelOutputLabels[0][cols - 1];
    debugPrint("O_O cols: $cols");

    if (glucoseSupply > 0) {
      supplyArr.add("glucose");
    }
    final hydrationSupply = modelOutputLabels[2][cols - 1];
    if (hydrationSupply > 0) {
      supplyArr.add("electrolyte water");
    }
    final proteinSupply = modelOutputLabels[3][cols - 1];
    if (proteinSupply > 0) {
      supplyArr.add("protein");
    }
    var muscleFatigueAdvise = "";
    final muscleFatigueSupply = modelOutputLabels[4][cols - 1];
    if (muscleFatigueSupply > 0) {
      muscleFatigueAdvise = "rest appropriately";
    }
    if (supplyArr.isEmpty && muscleFatigueAdvise.isEmpty) {
      return "* Supply advise: No suggestions for now.";
    }

    if (supplyArr.length == 1) {
      advise += "Supplement with ${supplyArr[0]}";
    } else if (supplyArr.length == 2) {
      advise += "Supplement with ${supplyArr[0]} and ${supplyArr[1]}";
    } else if (supplyArr.length == 3) {
      advise +=
          "Supplement with ${supplyArr[0]}, ${supplyArr[1]} and ${supplyArr[2]}";
    }
    if (muscleFatigueAdvise.isEmpty) {
      return "$advise.";
    } else {
      if (supplyArr.isEmpty) {
        advise += "$muscleFatigueAdvise.";
      } else {
        advise += ", and $muscleFatigueAdvise.";
      }
    }
    return advise;
  }
}