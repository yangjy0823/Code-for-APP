class Result {
  int code;
  String? msg;
  String? message;
  dynamic data;

  Result({required this.code, this.msg, this.data});

  factory Result.fromJSON(Map<String, dynamic> json) {
    return Result(code: json['code'], msg: json['message'], data: json['data']);
  }
}