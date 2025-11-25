class MdfMefBean {
  double mdf;
  double mef;

  MdfMefBean({required this.mdf, required this.mef});

  factory MdfMefBean.fromJSON(Map<String, dynamic> json) {
    return MdfMefBean(mdf: json['mdf'], mef: json['mef']);
  }
}

class MdfMefResult {
  int code;
  String? message;
  List<MdfMefBean>? data;
  int? total;
  int? page;
  int? pageSize;

  MdfMefResult({
    required this.code,
    this.message,
    this.data,
    this.total,
    this.page,
    this.pageSize,
  });

  factory MdfMefResult.fromJSON(Map<String, dynamic> json) {
    return MdfMefResult(
      code: json['code'],
      message: json['message'],
      data: (json['data'] as List?)
          ?.map((item) => MdfMefBean.fromJSON(item))
          .toList(),
      total: json['total'],
      page: json['page'],
      pageSize: json['page_size'],
    );
  }
}

class MftgBean {
  int fatigueClass;
  int glucoseClass;
  int hydrationClass;
  int lactateClass;
  int muscleFatigueClass;
  int proteinSupplyClass;
  int xTime;

  MftgBean({
    required this.fatigueClass,
    required this.glucoseClass,
    required this.hydrationClass,
    required this.lactateClass,
    required this.muscleFatigueClass,
    required this.proteinSupplyClass,
    required this.xTime,
  });

  factory MftgBean.fromJSON(Map<String, dynamic> json) {
    var xTimeValue = json['x_time'];
    int xTimeInt;

    if (xTimeValue is String) {
      xTimeInt = int.parse(xTimeValue);
    } else if (xTimeValue is int) {
      xTimeInt = xTimeValue;
    } else {
      xTimeInt = DateTime.now().millisecondsSinceEpoch;
    }

    return MftgBean(
      fatigueClass: json['fatigue_class'],
      glucoseClass: json['glucose_class'],
      hydrationClass: json['hydration_class'],
      lactateClass: json['lactate_class'],
      muscleFatigueClass: json['muscle_fatigue_class'],
      proteinSupplyClass: json['protein_supply_class'],
      xTime: xTimeInt,
    );
  }
}

class MftgResult {
  int code;
  String? message;
  List<MftgBean>? data;
  int? total;
  int? page;
  int? pageSize;

  MftgResult({
    required this.code,
    this.message,
    this.data,
    this.total,
    this.page,
    this.pageSize,
  });

  factory MftgResult.fromJSON(Map<String, dynamic> json) {
    return MftgResult(
      code: json['code'],
      message: json['message'],
      data: (json['data'] as List?)
          ?.map((item) => MftgBean.fromJSON(item))
          .toList(),
      total: json['total'],
      page: json['page'],
      pageSize: json['page_size'],
    );
  }
}