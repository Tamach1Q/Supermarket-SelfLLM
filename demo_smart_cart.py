import argparse
import json
import os
import re
import subprocess
import sys
import time
import tempfile
import wave
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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

    # 3) そこそこ近いもの（rapidfuzz優先、なければdifflib）
    try:
        from rapidfuzz import fuzz  # type: ignore
    except Exception:
        fuzz = None

    if fuzz is not None:
        scored: List[Tuple[int, Dict]] = []
        for it in db["items"]:
            score = fuzz.ratio(target, _norm(it["name"]))
            scored.append((score, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored and scored[0][0] >= 72:
            best = scored[0][1]
    else:
        import difflib

        candidates = [_norm(it["name"]) for it in db["items"]]
        close = difflib.get_close_matches(target, candidates, n=1, cutoff=0.72)
        if close:
            idx = candidates.index(close[0])
            best = db["items"][idx]

    return best

def _save_wav_pcm16_mono(path: str, pcm16: bytes, sample_rate: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16)


def _safe_print_err(msg: str) -> None:
    print(msg, file=sys.stderr)


@dataclass
class VADConfig:
    sample_rate: int = 16000
    frame_ms: int = 20
    aggressiveness: int = 2  # 0-3
    padding_ms: int = 300  # 開始検出用のリングバッファ
    start_ratio: float = 0.6  # padding内で音声と判定される割合で開始
    end_silence_ms: int = 600  # この無音で終了
    min_utterance_ms: int = 300  # 短すぎる発話は捨てる
    max_utterance_s: float = 6.0  # 取りこぼし防止の上限


class VADRecorder:
    def __init__(self, cfg: VADConfig, input_device: Optional[int] = None):
        self.cfg = cfg
        self.input_device = input_device

    def record_utterance(self) -> bytes:
        try:
            import collections

            import sounddevice as sd  # type: ignore
            import webrtcvad  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "VAD録音に必要な依存が不足しています。`pip install webrtcvad sounddevice numpy` を導入してください。"
            ) from e

        sample_rate = self.cfg.sample_rate
        frame_samples = int(sample_rate * self.cfg.frame_ms / 1000)
        if frame_samples * 1000 != sample_rate * self.cfg.frame_ms:
            raise ValueError("frame_msはsample_rateに対して割り切れる値にしてください（例: 10/20/30ms）。")

        vad = webrtcvad.Vad(self.cfg.aggressiveness)
        num_padding_frames = max(1, int(self.cfg.padding_ms / self.cfg.frame_ms))
        end_silence_frames = max(1, int(self.cfg.end_silence_ms / self.cfg.frame_ms))
        min_frames = max(1, int(self.cfg.min_utterance_ms / self.cfg.frame_ms))
        max_frames = max(1, int(self.cfg.max_utterance_s * 1000 / self.cfg.frame_ms))

        ring_buffer: "collections.deque[Tuple[bytes, bool]]" = collections.deque(maxlen=num_padding_frames)
        voiced_frames: List[bytes] = []
        triggered = False
        silent_run = 0

        with sd.RawInputStream(
            samplerate=sample_rate,
            blocksize=frame_samples,
            dtype="int16",
            channels=1,
            device=self.input_device,
        ) as stream:
            start_time = time.time()
            while True:
                if time.time() - start_time > (self.cfg.max_utterance_s + 5.0):
                    break

                frame, _overflowed = stream.read(frame_samples)
                if not frame:
                    continue

                is_speech = bool(vad.is_speech(frame, sample_rate))

                if not triggered:
                    ring_buffer.append((frame, is_speech))
                    num_voiced = sum(1 for _, speech in ring_buffer if speech)
                    if num_voiced >= int(self.cfg.start_ratio * ring_buffer.maxlen):
                        triggered = True
                        voiced_frames.extend(f for f, _ in ring_buffer)
                        ring_buffer.clear()
                else:
                    voiced_frames.append(frame)
                    silent_run = silent_run + 1 if not is_speech else 0
                    if silent_run >= end_silence_frames:
                        break
                    if len(voiced_frames) >= max_frames:
                        break

        if len(voiced_frames) < min_frames:
            return b""
        return b"".join(voiced_frames)


class FasterWhisperASR:
    def __init__(self, model_size_or_path: str, device: str, compute_type: str):
        try:
            import numpy as np  # type: ignore
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "ASRに必要な依存が不足しています。`pip install faster-whisper numpy` を導入してください。"
            ) from e
        self._np = np
        self.model = WhisperModel(model_size_or_path, device=device, compute_type=compute_type)

    def transcribe(self, pcm16: bytes, sample_rate: int) -> str:
        audio = self._np.frombuffer(pcm16, dtype=self._np.int16).astype(self._np.float32) / 32768.0
        segments, _info = self.model.transcribe(
            audio,
            language="ja",
            vad_filter=False,  # ここはwebrtcvad側でやる
            beam_size=1,
            temperature=0.0,
        )
        text = "".join(seg.text for seg in segments).strip()
        return re.sub(r"\s+", " ", text)


class LlamaCppLLM:
    def __init__(self, model_path: str, n_gpu_layers: int, n_ctx: int, chat_format: Optional[str]):
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "LLMに必要な依存が不足しています。`pip install llama-cpp-python` を導入してください。"
            ) from e

        kwargs = {
            "model_path": model_path,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "verbose": False,
        }
        if chat_format:
            kwargs["chat_format"] = chat_format
        self.llm = Llama(**kwargs)

    def chat_json(self, system: str, user: str, max_tokens: int) -> Dict:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        out = self.llm.create_chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        content = out["choices"][0]["message"]["content"]
        return _extract_json_object(content)


class CommandTTS:
    def __init__(self, command: List[str]):
        self.command = command

    def speak(self, text: str) -> None:
        text = re.sub(r"[\r\n]+", " ", text).strip()
        if not text:
            return
        try:
            subprocess.run(self.command, input=text, text=True, check=False)
        except Exception:
            pass


class NoTTS:
    def speak(self, text: str) -> None:
        text = re.sub(r"[\r\n]+", " ", text).strip()
        if text:
            print(f"\n[AIの応答]: {text}\n")


class OpenJTalkTTS:
    def __init__(
        self,
        open_jtalk_path: str,
        dict_path: str,
        voice_path: str,
        speed: float,
        aplay_path: str,
        aplay_device: Optional[str],
    ):
        self.open_jtalk_path = open_jtalk_path
        self.dict_path = dict_path
        self.voice_path = voice_path
        self.speed = speed
        self.aplay_path = aplay_path
        self.aplay_device = aplay_device

    def speak(self, text: str) -> None:
        text = re.sub(r"[\r\n]+", " ", text).strip()
        if not text:
            return
        wav_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(prefix="tts_", suffix=".wav", delete=False) as f:
                wav_path = f.name
            cmd = [
                self.open_jtalk_path,
                "-x",
                self.dict_path,
                "-m",
                self.voice_path,
                "-r",
                str(self.speed),
                "-ow",
                wav_path,
            ]
            subprocess.run(cmd, input=text, text=True, check=False)
            play_cmd = [self.aplay_path]
            if self.aplay_device:
                play_cmd += ["-D", self.aplay_device]
            play_cmd += ["-q", wav_path]
            subprocess.run(play_cmd, check=False)
        except Exception:
            pass
        finally:
            if wav_path:
                try:
                    os.unlink(wav_path)
                except Exception:
                    pass


def _extract_json_object(text: str) -> Dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("LLM出力からJSONが抽出できませんでした。")
    return json.loads(m.group(0))


def retrieve_candidate_products(query: str, db: Dict, top_k: int = 25) -> List[Dict]:
    q = _norm(query)
    if not q:
        return []
    try:
        from rapidfuzz import fuzz  # type: ignore
    except Exception:
        fuzz = None

    scored: List[Tuple[int, Dict]] = []
    for it in db.get("items", []):
        name = it.get("name", "")
        kws = [name] + it.get("keywords", [])
        best = 0
        for kw in kws:
            kw_n = _norm(str(kw))
            if not kw_n:
                continue
            if fuzz is not None:
                score = max(
                    fuzz.partial_ratio(q, kw_n),
                    fuzz.token_set_ratio(q, kw_n),
                )
            else:
                score = 100 if (kw_n in q or q in kw_n) else 0
            best = max(best, int(score))
        scored.append((best, it))

    scored.sort(key=lambda x: x[0], reverse=True)
    out = [it for score, it in scored if score > 0][:top_k]
    return out


def build_llm_prompts(transcript: str, cart_items: List[Dict], db: Dict, candidates: List[Dict]) -> Tuple[str, str]:
    db_names = [it.get("name", "") for it in db.get("items", []) if it.get("name")]
    cand_names = [it.get("name", "") for it in candidates if it.get("name")]
    cart_names = [it.get("name", "") for it in cart_items if it.get("name")]

    system = (
        "あなたは視覚障害者の買い物を支援する、スマートカートAIです。\n"
        "目的は会話ではなく、ユーザー発話を素早く『カートに入れる商品名リストの更新』に変換することです。\n"
        "出力は必ず次のJSONのみ（前後に説明・マークダウン禁止）:\n"
        '{"reply":"...", "items":[{"name":"...","quantity":"", "notes":""}]} \n'
        "\n"
        "制約:\n"
        "- items は『この時点のカート全体』。追加/削除を反映した最終結果を出す。\n"
        "- items[].name は、必ず DB_PRODUCTS に含まれる商品名のいずれかをそのまま使う（勝手に新しい名前を作らない）。\n"
        "- quantity/notes は不明なら空文字か省略可。\n"
        "- reply は1〜2文、です・ます調。\n"
        "  1文目: 反映した主要商品名を3〜6個、読点区切りで明示。\n"
        "  2文目: 追加確認を1つだけ（短く）。\n"
        "- 棚番号・価格・在庫には触れない。\n"
        "- 不確実なら items を変えず、確認質問を1つだけする。\n"
    )

    user = (
        f"TRANSCRIPT:\n{transcript}\n\n"
        f"CURRENT_CART:\n{json.dumps(cart_names, ensure_ascii=False)}\n\n"
        f"CANDIDATES (参考):\n{json.dumps(cand_names, ensure_ascii=False)}\n\n"
        f"DB_PRODUCTS (許可リスト):\n{json.dumps(db_names, ensure_ascii=False)}\n"
    )
    return system, user

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

class SmartCartSessionLocal:
    """完全オフライン（VAD+ASR+LLM+TTS）での多ターン対話セッション。"""

    def __init__(
        self,
        db: Dict,
        vad: VADRecorder,
        asr: FasterWhisperASR,
        llm: LlamaCppLLM,
        tts,
        max_turns: int = 10,
        save_audio: bool = True,
        candidates_top_k: int = 25,
        llm_max_tokens: int = 160,
    ):
        self.db = db
        self.vad = vad
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.max_turns = max_turns
        self.cart_items: List[Dict] = []
        self.save_audio = save_audio
        self.candidates_top_k = candidates_top_k
        self.llm_max_tokens = llm_max_tokens

    def _should_end(self, transcript: str, reply: str) -> bool:
        text = (transcript or "") + (reply or "")
        end_keywords = ["以上です", "以上で", "終わり", "もういい", "大丈夫です", "ありがとう", "会計", "決済", "チェックアウト"]
        return any(kw in text for kw in end_keywords)

    def run(self):
        print("=== スマートカート（ローカル）セッション開始 (Ctrl+C で終了) ===")
        try:
            for turn in range(1, self.max_turns + 1):
                prompt = "こんにちは。何をお探しですか？" if turn == 1 else "続けてどうされますか？"
                self.tts.speak(prompt)
                time.sleep(0.25)

                t0 = time.time()
                pcm16 = self.vad.record_utterance()
                t_vad = time.time() - t0
                if not pcm16:
                    self.tts.speak("すみません、聞き取れませんでした。もう一度お願いします。")
                    continue

                audio_filename = f"my_request_turn{turn}.wav"
                if self.save_audio:
                    _save_wav_pcm16_mono(audio_filename, pcm16, self.vad.cfg.sample_rate)

                t1 = time.time()
                transcript = self.asr.transcribe(pcm16, self.vad.cfg.sample_rate)
                t_asr = time.time() - t1

                candidates = retrieve_candidate_products(transcript, self.db, top_k=self.candidates_top_k)
                system, user = build_llm_prompts(transcript, self.cart_items, self.db, candidates)

                t2 = time.time()
                result = self.llm.chat_json(system=system, user=user, max_tokens=self.llm_max_tokens)
                t_llm = time.time() - t2

                reply = str(result.get("reply", "")).strip()
                items = result.get("items", [])
                if not isinstance(items, list):
                    items = []

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

                payload = {
                    "transcript": transcript,
                    "reply": reply,
                    "items": items,
                }
                save_result(audio_filename, payload, matched)

                if reply:
                    self.tts.speak(reply)
                else:
                    self.tts.speak("反映しました。続けますか？")

                print(f"[timing] vad={t_vad:.3f}s asr={t_asr:.3f}s llm={t_llm:.3f}s total={(time.time()-t0):.3f}s")

                if self._should_end(transcript, reply):
                    print("=== セッションを終了します（終了発話を検出） ===")
                    break
        finally:
            print("=== スマートカート セッション終了 ===")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="スマートカート（完全オフライン: VAD+ASR+LLM+TTS）")
    p.add_argument("--products", default="products.json", help="商品DB(JSON)パス")
    p.add_argument("--max-turns", type=int, default=10)

    # VAD / Audio
    p.add_argument("--list-devices", action="store_true", help="入力デバイス一覧を表示して終了")
    p.add_argument("--input-device", type=int, default=None, help="sounddeviceの入力デバイス番号（未指定でデフォルト）")
    p.add_argument("--vad-aggr", type=int, default=2, choices=[0, 1, 2, 3])
    p.add_argument("--vad-padding-ms", type=int, default=300)
    p.add_argument("--vad-start-ratio", type=float, default=0.6)
    p.add_argument("--vad-end-silence-ms", type=int, default=600)
    p.add_argument("--vad-min-utterance-ms", type=int, default=300)
    p.add_argument("--vad-max-utterance-s", type=float, default=6.0)
    p.add_argument("--save-audio", action="store_true", help="各ターンのWAVを保存する")

    # ASR
    p.add_argument("--asr-model", default="small", help="faster-whisperのモデル名またはローカルパス")
    p.add_argument("--asr-device", default="cuda", help="cuda / cpu")
    p.add_argument("--asr-compute-type", default="int8_float16", help="例: int8_float16 / float16 / int8")

    # LLM
    p.add_argument("--llm-model", required=True, help="GGUFモデルパス（例: Qwen2.5-3B-Instruct Q4_K_M）")
    p.add_argument("--llm-n-gpu-layers", type=int, default=999)
    p.add_argument("--llm-n-ctx", type=int, default=2048)
    p.add_argument("--llm-chat-format", default=None, help="llama-cpp-pythonのchat_format（例: qwen / chatml 等）")
    p.add_argument("--llm-max-tokens", type=int, default=160)

    # Retrieval
    p.add_argument("--candidates-top-k", type=int, default=25)

    # TTS
    p.add_argument("--tts", choices=["none", "command", "openjtalk"], default="command")
    p.add_argument(
        "--tts-command",
        nargs="+",
        default=["espeak-ng", "-v", "ja", "--stdin"],
        help="TTSコマンド（標準入力でテキストを受ける想定）。例: espeak-ng -v ja --stdin",
    )
    p.add_argument("--openjtalk-path", default="open_jtalk")
    p.add_argument("--openjtalk-dict", default="/var/lib/mecab/dic/open-jtalk/naist-jdic")
    p.add_argument(
        "--openjtalk-voice",
        default="/usr/share/hts-voice/nitech-jp-atr503-m001/nitech_jp_atr503_m001.htsvoice",
    )
    p.add_argument("--openjtalk-speed", type=float, default=1.0)
    p.add_argument("--aplay-path", default="aplay")
    p.add_argument("--aplay-device", default=None, help="例: plughw:1,0（未指定ならデフォルト出力）")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    if args.list_devices:
        try:
            import sounddevice as sd  # type: ignore

            print(sd.query_devices())
            return 0
        except Exception as e:
            _safe_print_err(f"デバイス一覧の取得に失敗しました: {e}")
            _safe_print_err("`pip install sounddevice` の導入と、OS側のオーディオ設定を確認してください。")
            return 2


    db = load_products_db(args.products)
    if not db.get("items"):
        _safe_print_err("警告: 商品DBが空です。`products.json`を確認してください。")

    vad_cfg = VADConfig(
        aggressiveness=args.vad_aggr,
        padding_ms=args.vad_padding_ms,
        start_ratio=args.vad_start_ratio,
        end_silence_ms=args.vad_end_silence_ms,
        min_utterance_ms=args.vad_min_utterance_ms,
        max_utterance_s=args.vad_max_utterance_s,
    )
    vad = VADRecorder(vad_cfg, input_device=args.input_device)
    asr = FasterWhisperASR(args.asr_model, device=args.asr_device, compute_type=args.asr_compute_type)
    llm = LlamaCppLLM(
        model_path=args.llm_model,
        n_gpu_layers=args.llm_n_gpu_layers,
        n_ctx=args.llm_n_ctx,
        chat_format=args.llm_chat_format,
    )
    if args.tts == "none":
        tts = NoTTS()
    elif args.tts == "openjtalk":
        tts = OpenJTalkTTS(
            open_jtalk_path=args.openjtalk_path,
            dict_path=args.openjtalk_dict,
            voice_path=args.openjtalk_voice,
            speed=float(args.openjtalk_speed),
            aplay_path=args.aplay_path,
            aplay_device=args.aplay_device,
        )
    else:
        tts = CommandTTS(args.tts_command)

    session = SmartCartSessionLocal(
        db=db,
        vad=vad,
        asr=asr,
        llm=llm,
        tts=tts,
        max_turns=args.max_turns,
        save_audio=bool(args.save_audio),
        candidates_top_k=args.candidates_top_k,
        llm_max_tokens=args.llm_max_tokens,
    )
    session.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

