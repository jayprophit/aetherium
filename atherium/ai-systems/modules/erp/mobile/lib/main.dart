// ERP Mobile App Entry (Integrated)
import 'package:flutter/material.dart';

void main() => runApp(const ERPApp());

class ERPApp extends StatelessWidget {
  const ERPApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ERP (Knowledge-Base)',
      home: Scaffold(
        appBar: AppBar(title: const Text('ERP (Knowledge-Base)')),
        body: const Center(child: Text('Welcome to ERP Mobile!')),
      ),
    );
  }
}
