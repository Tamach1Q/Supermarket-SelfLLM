# Supermarket-SelfLLM

🛒 **AI搭載のスマートショッピングカート実装例**

スーパーマーケット向けの、ショッピングカートにアタッチメントとして装着する、スマートカートシステムです。Google Generative AI と音声認識を活用して、商品をインテリジェントに認識・マッチングします。

## 📋 プロジェクト概要

このプロジェクトは、以下の機能を備えたスマートショッピングカートシステムの実装例です：

- **AI駆動の商品認識**: Google Generative AI を使った自然言語処理
- **音声インターフェース**: Whisper による日本語音声認識
- **商品データベースマッチング**: 曖昧な商品名から正確な商品情報を検索
- **エッジデバイス対応**: Jetson Tx2にて実装しました。

## 🚀 機能

- 自然言語による商品検索
- 複数近似マッチングアルゴリズム（完全一致 → キーワード包含 → ファジーマッチング）
- リアルタイム商品DBの読み込みと更新
- 音声入力対応

## 📂 ファイル構成

```
.
├── cart_ver.1.py              # メインの実装例
├── demo_smart_cart.py         # デモンストレーション用スクリプト
├── products.json              # 商品データベース
├── requirements-jetson.txt    # 依存パッケージ（Jetson対応）
├── my_request.result.json     # リクエスト結果サンプル
├── my_request.transcript.txt  # トランスクリプトサンプル
└── README.md                  # このファイル
```

## 🔧 セットアップ

### 必要要件

- Python 3.8+
- Google API キー（Cloud の生成AI API を有効化）
- 対応OS: Linux（Jetson Nano推奨）、macOS、Windows

### インストール

1. リポジトリをクローン

```bash
git clone https://github.com/Tamach1Q/Supermarket-SelfLLM.git
cd Supermarket-SelfLLM
```

2. 依存パッケージをインストール

```bash
pip install -r requirements-jetson.txt
```

3. Google API キーを環境変数に設定

```bash
export GOOGLE_API_KEY="your_api_key_here"
```

## 💻 使用方法

### 基本的な実行

```bash
python demo_smart_cart.py
```

### カスタマイズ

`products.json` に商品情報を追加することで、認識できる商品を拡張できます。

```json
{
  "items": [
    {
      "name": "商品名",
      "price": 100,
      "category": "カテゴリ",
      "keywords": ["別名1", "別名2"]
    }
  ]
}
```

## 📦 依存パッケージ

| パッケージ | 用途 |
|-----------|------|
| `google-generativeai` | AI モデルとの通信 |
| `faster-whisper` | 日本語音声認識 |
| `sounddevice` | オーディオ入力 |
| `webrtcvad` | 音声検出 |
| `rapidfuzz` | ファジーマッチング |

## 📝 ライセンス

MIT License

## 👤 作成者

Tamach1Q / Organization

## 🤝 貢献

プルリクエストを歓迎します。大きな変更の場合は、まず Issue を開いて変更内容を議論してください。

## 📞 サポート

問題が発生した場合は、[GitHub Issues](https://github.com/Tamach1Q/Supermarket-SelfLLM/issues) で報告してください。

---

**開発環境**: Jetson Tx2 / Python 3.8+  
**最終更新**: 2026年3月
