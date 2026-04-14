#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_manifest_single_file.py
✅ إنشاء manifest ← في ملف واحد مباشر (بدون ملفات مؤقتة)
✅ WAV + TXT ← النصوص من ملفات TXT
✅ MioCodec ← للتوكنز فقط
✅ حفظ تلقائي ← كل 500 عينة
✅ قابل للاستئناف ← من آخر نقطة
"""
import json, os, torch, soundfile as sf, glob
from pathlib import Path
from datetime import datetime
import librosa
import sys

# ===== CONFIG =====
OUTPUT_MANIFEST = "data/manifests/full_wav_txt.json"
CODEC_PATH = "Aratako/MioCodec-25Hz-24kHz"
OFFSET = 70145
TARGET_SR = 24000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_EVERY = 500  # حفظ كل 500 عينة

# المجلدات المستهدفة
AUDIO_DIRS = [
    "/WAV/1/wav_24k",      #غير مكان ملفات الصوت
]

print("="*70)
print("🔄 إنشاء manifest ← ملف واحد مباشر")
print("="*70)
print(f"📊 Output: {OUTPUT_MANIFEST}")
print(f"💾 Save every: {SAVE_EVERY} samples")
print(f"🔄 Resume: Supported")
print("="*70)

# ===== 1. تحميل MioCodec =====
print(f"\n🔄 Loading MioCodec...")
from miocodec import MioCodecModel
try:
    codec = MioCodecModel.from_pretrained(
        repo_id=None,
        config_path=os.path.join(CODEC_PATH, "config.yaml"),
        weights_path=os.path.join(CODEC_PATH, "model.safetensors")
    ).eval().to(DEVICE)
    print(f"✅ Codec loaded on {DEVICE}")
except Exception as e:
    print(f"❌ Error loading codec: {e}")
    sys.exit(1)

# ===== 2. تحميل بيانات سابقة ← للاستئناف =====
manifest_data = []
start_index = 0

if os.path.exists(OUTPUT_MANIFEST):
    print(f"\n📖 Found existing manifest ← Resuming...")
    try:
        with open(OUTPUT_MANIFEST, 'r', encoding='utf-8') as f:
            manifest_data = json.load(f)
        start_index = len(manifest_data)
        print(f"   ✅ Resuming from sample {start_index + 1}")
    except Exception as e:
        print(f"   ⚠️  Error loading existing manifest: {e}")
        manifest_data = []
        start_index = 0

# ===== 3. جمع كل الملفات WAV و TXT =====
print(f"\n🔍 Collecting WAV and TXT files...")
wav_files = {}
txt_files = {}

for audio_dir in AUDIO_DIRS:
    if not os.path.exists(audio_dir):
        print(f"   ⚠️  Directory not found: {audio_dir}")
        continue
    
    dir_name = os.path.basename(audio_dir)
    print(f"\n📁 Scanning: {audio_dir}")
    
    # جمع WAV
    for wav_path in Path(audio_dir).glob("*.wav"):
        wav_name = wav_path.stem
        wav_files[f"{dir_name}/{wav_name}"] = str(wav_path)
    
    # جمع TXT
    for txt_path in Path(audio_dir).glob("*.txt"):
        txt_name = txt_path.stem
        txt_files[f"{dir_name}/{txt_name}"] = str(txt_path)
    
    wav_count = len([w for w in wav_files if w.startswith(dir_name)])
    txt_count = len([t for t in txt_files if t.startswith(dir_name)])
    print(f"   WAV: {wav_count} files | TXT: {txt_count} files")

print(f"\n📊 Total WAV files: {len(wav_files):,}")
print(f"📊 Total TXT files: {len(txt_files):,}")

# ===== 4. مطابقة WAV مع TXT =====
print(f"\n🔍 Matching WAV with TXT...")
matched_pairs = []

for key, wav_path in wav_files.items():
    if key in txt_files:
        matched_pairs.append({
            'key': key,
            'wav': wav_path,
            'txt': txt_files[key]
        })

print(f"   ✅ Matched pairs: {len(matched_pairs):,}")

# تخطي الملفات التي تم معالجتها
if start_index > 0 and start_index < len(matched_pairs):
    matched_pairs = matched_pairs[start_index:]
    print(f"   ⏭️  Skipped {start_index} already processed files")

print(f"   🎯 Remaining: {len(matched_pairs):,} files to process")

# ===== 5. معالجة كل ملف =====
print(f"\n🔄 Processing files...")
success_count = len(manifest_data)
error_count = 0
errors = []
last_save_index = len(manifest_data)

for i, pair in enumerate(matched_pairs, 1):
    try:
        wav_path = pair['wav']
        txt_path = pair['txt']
        filename = os.path.basename(wav_path)
        dir_name = os.path.basename(os.path.dirname(wav_path))
        
        # ----- قراءة النص من TXT -----
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        if not text:
            print(f"\n[{i + start_index}/{len(matched_pairs) + start_index}] ⚠️  Empty TXT: {filename}")
            error_count += 1
            continue
        
        # ----- قراءة الصوت -----
        data, sr = sf.read(wav_path, dtype='float32')
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        if sr != TARGET_SR:
            data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR, res_type='kaiser_best')
        
        duration = len(data) / TARGET_SR
        
        # تخطي الملفات غير الصالحة
        if duration < 0.5 or duration > 30:
            continue
        
        print(f"\n[{i + start_index}/{len(matched_pairs) + start_index}] 📝 {filename}")
        print(f"   📄 Text: {text[:60]}...")
        print(f"   ⏱️ Duration: {duration:.2f}s")
        
        # ----- MioCodec للتوكنز -----
        print(f"   🔄 Encoding with MioCodec...")
        
        audio_tensor = torch.from_numpy(data).float().unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            encoded = codec.encode(audio_tensor)
        
        # استخراج التوكنز
        content_tokens = encoded.content_token_indices
        tokens_raw = content_tokens.cpu().numpy()
        tokens_with_offset = tokens_raw + OFFSET
        
        audio_tokens = tokens_with_offset.tolist()
        token_count = len(audio_tokens)
        
        print(f"   🔢 Tokens: {token_count} | Range: [{min(audio_tokens)}, {max(audio_tokens)}]")
        
        # ----- بناء العينة -----
        sample = {
            "text": text,
            "audio": wav_path,
            "duration": round(duration, 2),
            "audio_tokens": audio_tokens,
            "token_count": token_count,
            "locale": "ar",
            "speaker_id": "arabic_speaker",
            "has_tashkeel": False,
            "tier": 1,
            "quality_score": 0.75,
            "quality_reasons": ["wav_txt_matched"],
            "source_dir": dir_name,
        }
        
        manifest_data.append(sample)
        success_count += 1
        
        print(f"   ✅ Success: {success_count}/{i + start_index}")
        
        # ----- حفظ كل SAVE_EVERY عينة -----
        if len(manifest_data) % SAVE_EVERY == 0 and len(manifest_data) != last_save_index:
            print(f"\n   💾 Saving progress... ({len(manifest_data):,} samples)")
            with open(OUTPUT_MANIFEST, 'w', encoding='utf-8') as f:
                json.dump(manifest_data, f, ensure_ascii=False, indent=2)
            last_save_index = len(manifest_data)
            print(f"   ✅ Saved: {OUTPUT_MANIFEST}")
            print(f"   📏 Size: {os.path.getsize(OUTPUT_MANIFEST)/1024/1024:.2f} MB")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Interrupted by user!")
        print(f"💾 Saving progress before exit...")
        with open(OUTPUT_MANIFEST, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved: {len(manifest_data):,} samples")
        sys.exit(0)
        
    except Exception as e:
        error_count += 1
        error_msg = f"{filename}: {str(e)}"
        errors.append(error_msg)
        print(f"   ❌ Error: {error_msg}")
        continue

# ===== 6. حفظ نهائي =====
print(f"\n💾 Saving final manifest...")
with open(OUTPUT_MANIFEST, 'w', encoding='utf-8') as f:
    json.dump(manifest_data, f, ensure_ascii=False, indent=2)

print(f"   ✅ Saved: {OUTPUT_MANIFEST}")
print(f"   📏 Size: {os.path.getsize(OUTPUT_MANIFEST)/1024/1024:.2f} MB")

# ===== 7. إحصائيات =====
print("\n" + "="*70)
print("📊 إحصائيات manifest النهائي")
print("="*70)

total_duration = sum(s['duration'] for s in manifest_data)
total_tokens = sum(s['token_count'] for s in manifest_data)

print(f"\n📈 الإجمالي:")
print(f"   - Total Samples: {len(manifest_data):,}")
print(f"   - Total Duration: {total_duration/3600:.2f} hours")
print(f"   - Total Tokens: {total_tokens:,}")
print(f"   - Average Duration: {total_duration/len(manifest_data):.2f}s")
print(f"   - Average Tokens: {total_tokens/len(manifest_data):.1f}")
print(f"   - Success Rate: {success_count}/{success_count+error_count} ({success_count/(success_count+error_count)*100:.1f}%)")
print(f"   - File Size: {os.path.getsize(OUTPUT_MANIFEST)/1024/1024:.2f} MB")

# حسب المجلد
print(f"\n📁 العينات حسب المجلد:")
dir_counts = {}
for s in manifest_data:
    dir_name = s.get('source_dir', 'unknown')
    dir_counts[dir_name] = dir_counts.get(dir_name, 0) + 1

for dir_name, count in sorted(dir_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"   - {dir_name}: {count:,} samples")

# 8. حفظ تقرير
print("\n" + "="*70)
report_path = OUTPUT_MANIFEST.replace('.json', '_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("Final Manifest Report (Single File)\n")
    f.write("="*50 + "\n\n")
    f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Output: {OUTPUT_MANIFEST}\n\n")
    f.write(f"Total Samples: {len(manifest_data):,}\n")
    f.write(f"Total Duration: {total_duration/3600:.2f} hours\n")
    f.write(f"Total Tokens: {total_tokens:,}\n")
    f.write(f"Success Rate: {success_count}/{success_count+error_count}\n")
    f.write(f"File Size: {os.path.getsize(OUTPUT_MANIFEST)/1024/1024:.2f} MB\n\n")
    f.write(f"Samples per directory:\n")
    for dir_name, count in sorted(dir_counts.items(), key=lambda x: x[1], reverse=True):
        f.write(f"   - {dir_name}: {count:,}\n")

print(f"✅ Report saved: {report_path}")
print("\n🎯 جاهز للتدريب ← ملف واحد نهائي!")
print("="*70)
