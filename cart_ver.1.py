import google.generativeai as genai
import os, sys, time, json, difflib, re
from typing import Dict, List, Optional

# --- 0. APIキー ---
try:
    GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
except KeyError:
    print("エラー: GOOGLE_API_KEYが環境変数に設定されていません。")
    sys.exit(1)

genai.configure(api_key=GOOGLE_API_KEY)

# --- DB読み込み ---
def load_products_db(path="products.json") -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            db = json.load(f)
        assert "items" in db and isinstance(db["items"], list)
        return db
    except Exception as e:
        print(f"商品DBの読み込みに失敗: {e}")
        return {"items": []}

PRODUCTS_DB = load_products_db()

# --- 簡易マッチング ---
def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s.replace("　", "")).lower()

def match_product(name: str, db: Dict) -> Optional[Dict]:
    """name（モデルが返した食材名）をDBのitemにマッチ。完全一致→キーワード包含→ゆるい近似の順。"""
    target = _norm(name)
    best = None

    # 1) 完全一致（name対name）
    for it in db["items"]:
        if _norm(it["name"]) == target:
            return it

    # 2) キーワード包含（どちらの包含も許す）
    for it in db["items"]:
        kws = [it["name"]] + it.get("keywords", [])
        for kw in kws:
            k = _norm(kw)
            if not k:
                continue
            if k in target or target in k:
                return it

    # 3) difflibでそこそこ近いもの
    candidates = [_norm(it["name"]) for it in db["items"]]
    close = difflib.get_close_matches(target, candidates, n=1, cutoff=0.72)
    if close:
        idx = candidates.index(close[0])
        best = db["items"][idx]

    return best

# --- 1. 録音（ffmpeg/avfoundation） ---
def record_audio(filename="request.wav", duration=4, prompt_text: Optional[str] = "こんにちは。何をお探しですか？"):
    """1ターン分の音声を録音。
    prompt_text が None の場合は事前の音声プロンプトを省略する。
    """
    time.sleep(1.0)  # 回り込み防止を少し長めに
    cmd = f'ffmpeg -f avfoundation -i ":0" -ac 1 -ar 16000 -t {duration} -y "{filename}"'
    if prompt_text:
        os.system(f'say -v "Kyoko" "{prompt_text}"')
    print(f"実行中: {cmd}")
    os.system(cmd)
    os.system('say -v "Kyoko" "はい。回答を生成します。"')
    return filename

# --- 2. Geminiに投げる（JSON出力を強制） ---
def _build_context_text(history: Optional[List[Dict]], cart_items: Optional[List[Dict]]) -> str:
    """これまでの会話と現在のカート状態をテキスト化してLLMに渡す。"""
    lines: List[str] = []

    if history:
        lines.append("【これまでの会話履歴】")
        for idx, h in enumerate(history, 1):
            role = "ユーザー" if h.get("role") == "user" else "アシスタント"
            text = h.get("text", "").replace("\n", " ")
            lines.append(f"{idx}. {role}: {text}")

    if cart_items:
        lines.append("【現在のカートに入っている食材】")
        for it in cart_items:
            nm = it.get("name") or ""
            qty = it.get("quantity") or ""
            qty_str = f" x{qty}" if qty else ""
            lines.append(f"- {nm}{qty_str}")

    if lines:
        lines.append(
            "上記の履歴とカート状態を踏まえて、"
            "ユーザーの最新の音声発話時点でのカート全体を items に出力してください。"
        )

    return "\n".join(lines)


def get_llm_response(audio_file, history: Optional[List[Dict]] = None, cart_items: Optional[List[Dict]] = None):
    # システム指示（Few-shot最小例付き。出力はJSONのみ）
    system_prompt = """
あなたはスーパーマーケットのショッピングカートに搭載されたAIアシスタントです。
音声リクエストと、テキストで渡されるこれまでの会話・現在のカート状態を踏まえて、
最新の発話時点での「カートに入れるべき食材の全体リスト」を更新し、指定のJSONのみを出力します。

出力は必ず次のJSONのみ（前後に説明・マークダウン禁止）:
{"transcript":"...", "reply":"...", "items":[{"name":"..."}, ...]}

定義:
- transcript: 最新の音声から聞き取れた日本語をそのまま。誤認識も改変しない。
- items: この対話セッション全体で、現時点のカートに入っているべき食材の一覧。
  各要素は { "name": string, "quantity": string(任意), "notes": string(任意) }。
  量や補足が分からなければ quantity, notes は空文字か省略可。
  カートが空なら空配列。
- reply: 1〜2文、です・ます調。次を厳守。
  1) 1文目でユーザーの意図を言い直しつつ、itemsに含めた主要食材名を3〜6個、読点区切りで明示する
     （例: 焼きそば麺、キャベツ、ベーコン、焼きそばソースです。）。
  2) 2文目は短い確認または次アクションを1つだけ（人数確認、辛さ/脂質などの嗜好、代替可否など）。
     長々と説明しない。
  3) itemsに存在しない食材名を勝手に追加しない。replyの食材名は items[].name をそのまま用いる。
  4) 棚番号・価格・在庫は書かない（それはシステム側で付与する）。
  5) 汎用フレーズのみで終わらせない。禁止: 「承知しました。」だけ、「必要な食材をお探しします。」だけ。

複数ターンの更新:
- テキストで渡される「これまでの会話履歴」「現在のカート状態」を見て、
  最新の音声発話によってカートをどう更新すべきかを判断してください。
- 例: 直前のカートに「キャベツ」が含まれていて、最新の発話が
  「やっぱりキャベツはいらない」の場合、items からキャベツを除外した最終的なカート全体を出力する。

曖昧・聞き取り不良:
- transcriptに不明瞭さがあってもそのまま出力する。
- itemsを決められない場合は空配列にし、replyでは要点を1つだけ丁寧に確認する。

例（説明であり出力に含めない）:
1ターン目: 「焼きそば買いたい」
  → items: 焼きそば麺, キャベツ, 豚肉, 焼きそばソース
2ターン目: 「やっぱキャベツはいらない」
  → items: 焼きそば麺, 豚肉, 焼きそばソース

"""

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system_prompt
    )

    try:
        audio_file_g = genai.upload_file(path=audio_file, mime_type="audio/wav")
        print("--- アップロード完了 ---")
    except Exception as e:
        print(f"エラー: ファイルのアップロードに失敗しました。 {e}")
        return None

    generation_config = {
        "temperature": 0.1,
        "response_mime_type": "application/json",
        "response_schema": {
            "type": "object",
            "properties": {
                "transcript": {"type": "string"},
                "reply": {"type": "string"},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "quantity": {"type": "string"},
                            "notes": {"type": "string"}
                        },
                        "required": ["name"]
                    }
                }
            },
            "required": ["transcript", "reply", "items"]
        }
    }

    parts = []
    context_text = _build_context_text(history, cart_items)
    if context_text:
        parts.append(context_text)
    parts.append(audio_file_g)

    try:
        response = model.generate_content(parts, generation_config=generation_config)
        if not getattr(response, "text", None) or not response.text.strip():
            print("--- 応答テキストが空 ---")
            return None
        # 100%純JSONのはずだが、念のためフォールバック
        try:
            return json.loads(response.text)
        except Exception:
            m = re.search(r"\{.*\}", response.text, re.DOTALL)
            return json.loads(m.group(0)) if m else None
    except Exception as e:
        print("--- Gemini APIエラー ---")
        print(e)
        return None

# --- 3. 返答読み上げ ---
def speak_response(text):
    print("--- 音声出力 ---")
    safe_output = text.replace('"', '\\"')
    os.system(f'say -v "Kyoko" "{safe_output}"')
    print(f"\n[AIの応答]: {text}\n")

# --- 4. 保存 ---
def save_result(audio_file, result_obj, matched_items):
    try:
        base, _ = os.path.splitext(audio_file)
        # テキスト
        out_txt = f"{base}.transcript.txt"
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(out_txt, 'w', encoding='utf-8') as f:
            f.write(f"timestamp: {ts}\n")
            f.write(f"audio: {audio_file}\n\n")
            f.write("[文字起こし]\n")
            f.write(result_obj.get("transcript","") + "\n\n")
            f.write("[AIの応答]\n")
            f.write(result_obj.get("reply","") + "\n\n")
            f.write("[必要なアイテム]\n")
            for it in matched_items:
                shelf = it.get("shelf", "不明")
                nm = it.get("name")
                qty = it.get("quantity","")
                qty = f" x{qty}" if qty else ""
                f.write(f"- {nm}{qty} （棚 {shelf}）\n")
        print(f"保存: {out_txt}")

        # 構造化結果
        out_json = f"{base}.result.json"
        payload = {
            "timestamp": ts,
            "audio": audio_file,
            "transcript": result_obj.get("transcript",""),
            "reply": result_obj.get("reply",""),
            "items": result_obj.get("items", []),
            "matched_items": matched_items
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"保存: {out_json}")
    except Exception as e:
        print(f"保存に失敗: {e}")

# --- 5. マッチ結果の組み立て ---
def annotate_with_shelf(items: List[Dict], db: Dict) -> List[Dict]:
    out = []
    for it in items:
        nm = it.get("name","").strip()
        if not nm:
            continue
        m = match_product(nm, db)
        enriched = {
            "name": nm,
            "quantity": it.get("quantity",""),
            "notes": it.get("notes",""),
            "product_id": m.get("id") if m else None,
            "shelf": m.get("shelf") if m else None,
            "unit": m.get("unit") if m else None,
            "db_name": m.get("name") if m else None
        }
        out.append(enriched)
    return out

class SmartCartSession:
    """全体カート方式での多ターン対話セッション。"""

    def __init__(self, db: Dict, max_turns: int = 10):
        self.db = db
        self.max_turns = max_turns
        self.history: List[Dict] = []
        self.cart_items: List[Dict] = []

    def _should_end(self, transcript: str, reply: str) -> bool:
        text = (transcript or "") + (reply or "")
        end_keywords = ["以上です", "以上で", "終わり", "もういい", "大丈夫です", "ありがとう"]
        return any(kw in text for kw in end_keywords)

    def run(self):
        print("=== スマートカート セッション開始 (Ctrl+C で終了) ===")
        try:
            for turn in range(1, self.max_turns + 1):
                audio_filename = f"my_request_turn{turn}.wav"
                prompt = "こんにちは。何をお探しですか？" if turn == 1 else "続けてどうされますか？"
                audio_path = record_audio(filename=audio_filename, duration=4, prompt_text=prompt)

                if not audio_path:
                    break

                result = get_llm_response(audio_path, history=self.history, cart_items=self.cart_items)
                if not isinstance(result, dict):
                    speak_response("エラー。Gemini APIで応答が生成できませんでした。")
                    break

                transcript = result.get("transcript", "")
                reply = result.get("reply", "")
                items = result.get("items", [])

                print("\n=== 文字起こし ===")
                print(transcript)
                print("=================\n")

                matched = annotate_with_shelf(items, self.db)
                self.cart_items = matched

                if matched:
                    print("=== 現在のカート（棚付き） ===")
                    for it in matched:
                        shelf = it.get("shelf") or "不明"
                        qty = f' x{it["quantity"]}' if it.get("quantity") else ""
                        print(f"- {it['name']}{qty} | 棚: {shelf}")
                    print("============================\n")
                else:
                    print("カートは空です\n")

                save_result(audio_path, result, matched)

                self.history.append({"role": "user", "text": transcript})
                self.history.append({"role": "assistant", "text": reply})

                speak_response(reply if reply else "ご要望を認識しました。")

                if self._should_end(transcript, reply):
                    print("=== セッションを終了します（終了発話を検出） ===")
                    break
        finally:
            print("=== スマートカート セッション終了 ===")


# --- メイン ---
if __name__ == "__main__":
    session = SmartCartSession(PRODUCTS_DB)
    session.run()
