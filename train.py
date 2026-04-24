"""
╔══════════════════════════════════════════════════════════════════════╗
║   TRAINING SCRIPT — Label Studio YOLO ZIP → RT-DETR                ║
║                                                                      ║
║   Input  : file .zip hasil export Label Studio (format YOLO)        ║
║   Output : model best.pt siap dipakai di v.py                      ║
║                                                                      ║
║   Cara export di Label Studio:                                       ║
║   Project → Export → YOLO → Download ZIP                           ║
║                                                                      ║
║   Cara pakai script ini:                                             ║
║   1. Isi KONFIGURASI di bawah                                       ║
║   2. python train_labelstudio.py                                     ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os, sys, re, shutil, random, time, zipfile, yaml
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
#  KONFIGURASI — SESUAIKAN SEBELUM MENJALANKAN
# ═══════════════════════════════════════════════════════════════════

# Path ke file ZIP hasil export Label Studio
# Contoh: r"C:\Users\Anda\Downloads\project-1-at-2025-04-23.zip"
ZIP_PATH      = r"C:\Users\ASUS\Documents\LocObj\src\anotation-export\project-5-at-2026.zip"  # ← GANTI
# project-5-at-2026-04-23-21-48-f1a4314d.zip
# C:\Users\ASUS\Documents\LocObj\project-5-at-2026.zip


# Folder tujuan ekstrak dan persiapan dataset
# Script akan membuat folder ini secara otomatis
DATASET_DIR   = r"C:\Users\ASUS\Documents\LocObj\model\dataset_kendaraan"                               # ← GANTI

# Nama class yang Anda pakai saat anotasi di Label Studio
# Harus sama persis (case-sensitive) dengan nama label di Label Studio
# Satu class untuk semua kendaraan (belakang + depan)
CLASS_NAMES   = ['carback', 'carfront']                                    # ← GANTI jika berbeda

# ── Parameter training ────────────────────────
IMG_SIZE      = 640
EPOCHS        = 60
BATCH_SIZE    = 4        # 4GB VRAM→4 | 8GB→8 | 16GB→16
NAMA_TRAINING = "vehicle_labelstudio_v1"
BASE_MODEL    = "rtdetr-l.pt"   # model pretrained — didownload otomatis jika belum ada

# ── Split ratio ───────────────────────────────
RATIO_TRAIN   = 0.80   # 80% training
RATIO_VALID   = 0.10   # 10% validasi
# sisa 10% → test

# ═══════════════════════════════════════════════════════════════════
#  TERMINAL HELPER
# ═══════════════════════════════════════════════════════════════════
def c(text, color):
    codes = {
        'green':'\033[92m','yellow':'\033[93m','red':'\033[91m',
        'cyan':'\033[96m','white':'\033[97m','gray':'\033[90m',
        'bold':'\033[1m','reset':'\033[0m'
    }
    return codes.get(color,'')+str(text)+codes['reset']
def sep():    print(c('─'*66,'gray'))
def log(m,cl='white'): print(c(f"  {m}",cl))
def ok(m):    log(f"✓  {m}",'green')
def warn(m):  log(f"⚠  {m}",'yellow')
def info(m):  log(f"→  {m}",'cyan')
def err(m):   log(f"✗  {m}",'red'); sys.exit(1)
def hdr(m):   sep(); log(m,'bold'); sep()

# ═══════════════════════════════════════════════════════════════════
#  LANGKAH 1 — CEK GPU
# ═══════════════════════════════════════════════════════════════════
def check_gpu():
    hdr("LANGKAH 1 — CEK GPU")
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            ok(f"GPU    : {name}")
            ok(f"VRAM   : {vram:.1f} GB")
            sug  = 4 if vram < 5 else (8 if vram < 9 else 16)
            if BATCH_SIZE > sug:
                warn(f"BATCH_SIZE={BATCH_SIZE} mungkin terlalu besar untuk {vram:.0f} GB VRAM")
                warn(f"Disarankan BATCH_SIZE = {sug}")
            return 0   # GPU index
        else:
            warn("GPU tidak ditemukan — training pakai CPU (sangat lambat)")
            warn("Pertimbangkan Google Colab: colab.research.google.com")
            return 'cpu'
    except ImportError:
        warn("PyTorch belum terinstall:  pip install torch torchvision")
        return 'cpu'

# ═══════════════════════════════════════════════════════════════════
#  LANGKAH 2 — EKSTRAK ZIP LABEL STUDIO
# ═══════════════════════════════════════════════════════════════════
def extract_zip(zip_path: str, dataset_dir: Path) -> Path:
    hdr("LANGKAH 2 — EKSTRAK ZIP LABEL STUDIO")

    zp = Path(zip_path)
    if not zp.exists():
        err(f"File ZIP tidak ditemukan:\n     {zip_path}\n"
            "     Pastikan path sudah benar di bagian KONFIGURASI")
    if not zipfile.is_zipfile(zp):
        err(f"File bukan ZIP yang valid: {zip_path}")

    extract_dir = dataset_dir / "_raw_extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)

    ok(f"ZIP ditemukan  : {zp.name}  ({zp.stat().st_size/1e6:.1f} MB)")
    info(f"Ekstrak ke     : {extract_dir}")

    with zipfile.ZipFile(zp, 'r') as zf:
        members = zf.namelist()
        info(f"Jumlah file    : {len(members)}")
        zf.extractall(extract_dir)

    ok("Ekstrak selesai")
    return extract_dir

# ═══════════════════════════════════════════════════════════════════
#  LANGKAH 3 — ANALISIS STRUKTUR LABEL STUDIO YOLO
#
#  Label Studio YOLO ZIP bisa menghasilkan berbagai struktur:
#
#  Struktur A (paling umum — Label Studio terbaru):
#    images/
#      train/  *.jpg
#      val/    *.jpg
#    labels/
#      train/  *.txt
#      val/    *.txt
#    classes.txt
#    notes.json
#
#  Struktur B (flat):
#    *.jpg, *.txt, classes.txt  (semua di root)
#
#  Struktur C (sub-folder per class):
#    obj_train_data/
#      *.jpg, *.txt
#    obj_valid_data/
#      *.jpg, *.txt
#    obj.data
#    obj.names
#
#  Script ini mendeteksi dan menangani ketiga struktur otomatis.
# ═══════════════════════════════════════════════════════════════════
def detect_ls_structure(extract_dir: Path) -> str:
    """
    Deteksi struktur ZIP Label Studio.
    Returns: 'ls_split' | 'ls_flat' | 'ls_obj' | 'unknown'
    """
    # Struktur A: ada folder images/train dan labels/train
    if ((extract_dir/'images'/'train').exists() or
        (extract_dir/'images'/'val').exists()):
        return 'ls_split'

    # Struktur C: ada obj_train_data atau obj_valid_data
    obj_dirs = list(extract_dir.rglob('obj_train_data'))
    if obj_dirs:
        return 'ls_obj'

    # Struktur B: cari gambar di root atau satu level bawah
    imgs = list(extract_dir.rglob('*.jpg')) + list(extract_dir.rglob('*.png'))
    if imgs:
        return 'ls_flat'

    return 'unknown'

# ═══════════════════════════════════════════════════════════════════
#  LANGKAH 4 — BACA classes.txt / obj.names
# ═══════════════════════════════════════════════════════════════════
def read_classes(extract_dir: Path) -> list:
    """
    Baca nama class dari file classes.txt atau obj.names di ZIP.
    Returns daftar nama class sesuai urutan index.
    """
    candidates = (
        list(extract_dir.rglob('classes.txt')) +
        list(extract_dir.rglob('obj.names'))   +
        list(extract_dir.rglob('*.names'))
    )
    if not candidates:
        warn("File classes.txt / obj.names tidak ditemukan di ZIP")
        warn(f"Menggunakan CLASS_NAMES dari konfigurasi: {CLASS_NAMES}")
        return CLASS_NAMES

    cls_file = candidates[0]
    lines = [l.strip() for l in cls_file.read_text(encoding='utf-8').splitlines()
             if l.strip()]
    ok(f"Class dari ZIP : {lines}  (file: {cls_file.name})")

    # Validasi terhadap CLASS_NAMES di konfigurasi
    if lines != CLASS_NAMES:
        warn(f"Class di ZIP   : {lines}")
        warn(f"CLASS_NAMES    : {CLASS_NAMES}")
        warn("Perbedaan nama class — menggunakan class dari ZIP")
        return lines
    return lines

# ═══════════════════════════════════════════════════════════════════
#  LANGKAH 5 — NORMALISASI DATASET KE STRUKTUR YOLOV8
#
#  Target struktur yang diinginkan:
#    dataset_dir/
#      train/images/*.jpg
#      train/labels/*.txt
#      valid/images/*.jpg
#      valid/labels/*.txt
#      test/images/*.jpg
#      test/labels/*.txt
#      data.yaml
# ═══════════════════════════════════════════════════════════════════
def collect_pairs(extract_dir: Path) -> list:
    """
    Kumpulkan semua pasangan (gambar, label) dari extract_dir.
    Returns list of (img_path, lbl_path).
    """
    img_exts = {'.jpg','.jpeg','.png','.bmp','.webp'}
    pairs    = []
    all_imgs = []
    for ext in img_exts:
        all_imgs.extend(extract_dir.rglob(f'*{ext}'))

    for img in all_imgs:
        # Cari label .txt dengan nama stem sama
        lbl = img.with_suffix('.txt')
        if not lbl.exists():
            # Coba di folder labels/ sejajar dengan folder images/
            parts = img.parts
            for i, part in enumerate(parts):
                if part.lower() in ('images','image','imgs'):
                    lbl_parts = list(parts)
                    lbl_parts[i] = 'labels'
                    lbl = Path(*lbl_parts).with_suffix('.txt')
                    break
        if lbl.exists():
            pairs.append((img, lbl))
        else:
            warn(f"Label tidak ada untuk: {img.name} — dilewati")

    return pairs


def normalize_dataset(extract_dir: Path, dataset_dir: Path, classes: list) -> Path:
    hdr("LANGKAH 5 — NORMALISASI STRUKTUR DATASET")

    structure = detect_ls_structure(extract_dir)
    info(f"Struktur ZIP terdeteksi: {structure}")

    # ── Kasus struktur A: sudah displit Label Studio ──────────────
    if structure == 'ls_split':
        info("Label Studio sudah membagi train/val — mapping ke train/valid/test")
        pairs_train, pairs_val = [], []

        for split_src, split_dst in [('train','train'),('val','valid')]:
            img_dir = extract_dir/'images'/split_src
            lbl_dir = extract_dir/'labels'/split_src
            if not img_dir.exists(): continue
            for img in img_dir.iterdir():
                if img.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}:
                    lbl = lbl_dir/(img.stem+'.txt') if lbl_dir.exists() else img.with_suffix('.txt')
                    if lbl.exists():
                        (pairs_train if split_dst=='train' else pairs_val).append((img,lbl))
                    else:
                        warn(f"Label tidak ada: {img.name}")

        # Bagi pairs_val → 50% valid 50% test
        random.seed(42); random.shuffle(pairs_val)
        mid = max(1, len(pairs_val)//2)
        splits = {'train': pairs_train,
                  'valid': pairs_val[:mid],
                  'test':  pairs_val[mid:]}

    # ── Kasus struktur B/C: flat atau obj_*_data ──────────────────
    else:
        all_pairs = collect_pairs(extract_dir)
        if not all_pairs:
            err("Tidak ada pasangan gambar+label ditemukan di ZIP!\n"
                "     Pastikan Anda mengekspor format YOLO dari Label Studio.")
        random.seed(42); random.shuffle(all_pairs)
        n  = len(all_pairs)
        nt = max(1, int(n * RATIO_TRAIN))
        nv = max(1, int(n * RATIO_VALID))
        splits = {'train': all_pairs[:nt],
                  'valid': all_pairs[nt:nt+nv],
                  'test':  all_pairs[nt+nv:]}

    # ── Salin ke struktur YOLOv8 bersih ──────────────────────────
    total = sum(len(v) for v in splits.values())
    info(f"Total gambar berlabel: {total}")

    if total < 10:
        warn(f"Hanya {total} gambar — lanjutkan anotasi di Label Studio")
        warn("Jalankan script ini lagi setelah anotasi bertambah")

    for split_name, pairs in splits.items():
        out_img = dataset_dir/split_name/'images'
        out_lbl = dataset_dir/split_name/'labels'
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        for img, lbl in pairs:
            shutil.copy2(img, out_img/img.name)
            shutil.copy2(lbl, out_lbl/(img.stem+'.txt'))
        info(f"  {split_name:5s}: {len(pairs):4d} gambar ({len(pairs)/max(total,1)*100:.0f}%)")

    ok("Normalisasi selesai")
    return dataset_dir


def remap_label_indices(dataset_dir: Path, zip_classes: list, target_classes: list):
    """
    Jika urutan class di ZIP berbeda dengan CLASS_NAMES,
    remap index di semua file .txt secara otomatis.
    """
    if zip_classes == target_classes:
        return  # tidak perlu remap

    hdr("REMAP CLASS INDEX")
    warn(f"Class ZIP    : {zip_classes}")
    warn(f"Class target : {target_classes}")

    mapping = {}
    for new_idx, name in enumerate(target_classes):
        if name in zip_classes:
            old_idx = zip_classes.index(name)
            mapping[old_idx] = new_idx
            info(f"  index {old_idx} ({name}) → index {new_idx}")
        else:
            warn(f"  Class '{name}' tidak ada di ZIP — akan hilang dari label")

    for lbl_file in dataset_dir.rglob('*.txt'):
        if lbl_file.name == 'classes.txt': continue
        lines = lbl_file.read_text(encoding='utf-8').splitlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            old = int(parts[0])
            new = mapping.get(old)
            if new is None:
                warn(f"Index {old} tidak dikenali di {lbl_file.name} — baris dilewati")
                continue
            new_lines.append(f"{new} " + " ".join(parts[1:]))
        lbl_file.write_text("\n".join(new_lines)+"\n", encoding='utf-8')

    ok("Remap index selesai")

# ═══════════════════════════════════════════════════════════════════
#  LANGKAH 6 — VALIDASI LABEL
#  Cek format file .txt: pastikan koordinat dalam range 0.0–1.0
# ═══════════════════════════════════════════════════════════════════
def validate_labels(dataset_dir: Path, classes: list):
    hdr("LANGKAH 6 — VALIDASI LABEL")
    total = 0; errors = 0; empty = 0
    nc = len(classes)

    for lbl in dataset_dir.rglob('*.txt'):
        if lbl.name == 'classes.txt': continue
        lines = [l.strip() for l in lbl.read_text(encoding='utf-8').splitlines() if l.strip()]
        if not lines:
            empty += 1; continue
        for line in lines:
            parts = line.split()
            total += 1
            if len(parts) != 5:
                warn(f"Format salah (bukan 5 kolom): {lbl.name} → {line[:50]}")
                errors += 1; continue
            idx = int(parts[0])
            coords = list(map(float, parts[1:]))
            if idx >= nc:
                warn(f"Index class {idx} melebihi nc={nc}: {lbl.name}")
                errors += 1
            for v in coords:
                if not (0.0 <= v <= 1.0):
                    warn(f"Koordinat di luar 0–1: {lbl.name} → {line[:50]}")
                    errors += 1; break

    ok(f"Total anotasi   : {total}")
    info(f"Label kosong    : {empty} (background — normal)")
    if errors == 0:
        ok("Semua label valid ✓")
    else:
        warn(f"Ditemukan {errors} baris bermasalah — periksa anotasi di Label Studio")

# ═══════════════════════════════════════════════════════════════════
#  LANGKAH 7 — BUAT data.yaml
# ═══════════════════════════════════════════════════════════════════
def make_yaml(dataset_dir: Path, classes: list) -> Path:
    hdr("LANGKAH 7 — BUAT data.yaml")
    counts = {
        s: len(list((dataset_dir/s/'images').glob('*.*')))
           if (dataset_dir/s/'images').exists() else 0
        for s in ['train','valid','test']
    }
    data = {
        'path':  str(dataset_dir.resolve()),
        'train': 'train/images',
        'val':   'valid/images',
        'test':  'test/images',
        'nc':    len(classes),
        'names': classes,
    }
    yaml_path = dataset_dir/'data.yaml'
    with open(yaml_path,'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    ok(f"data.yaml → {yaml_path}")
    info(f"  train : {counts['train']} gambar")
    info(f"  valid : {counts['valid']} gambar")
    info(f"  test  : {counts['test']}  gambar")
    info(f"  nc    : {len(classes)}  →  {classes}")
    return yaml_path

# ═══════════════════════════════════════════════════════════════════
#  LANGKAH 8 — TRAINING
# ═══════════════════════════════════════════════════════════════════
def run_training(yaml_path: Path, device):
    hdr("LANGKAH 8 — TRAINING RT-DETR")
    try:
        from ultralytics import RTDETR
    except ImportError:
        err("Ultralytics tidak terinstall: pip install ultralytics")

    info(f"Base model   : {BASE_MODEL}")
    info(f"Dataset YAML : {yaml_path}")
    info(f"Epochs       : {EPOCHS}")
    info(f"Batch        : {BATCH_SIZE}")
    info(f"Img size     : {IMG_SIZE}×{IMG_SIZE}")
    info(f"Device       : {'GPU cuda:'+str(device) if device!='cpu' else 'CPU'}")
    info(f"Nama sesi    : {NAMA_TRAINING}")
    sep()
    log("Training dimulai...  Ctrl+C untuk stop", 'yellow')
    log(f"Checkpoint  → runs/detect/{NAMA_TRAINING}/weights/", 'cyan')
    print()

    existing = Path('runs')/'detect'/NAMA_TRAINING
    if existing.exists():
        warn(f"Folder '{NAMA_TRAINING}' sudah ada — Ultralytics akan tambah suffix otomatis")

    # ── Fix cuDNN crash (CUDA 12.8 + Python 3.14 compatibility) ─────
    import torch
    torch.backends.cudnn.enabled          = False  # nonaktifkan cuDNN → pakai CUDA fallback
    torch.backends.cudnn.benchmark        = False  # matikan auto-tuner
    torch.backends.cuda.matmul.allow_tf32 = False  # matikan TF32
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32   = False
    ok("cuDNN dinonaktifkan — pakai CUDA fallback (stabil untuk CUDA 12.8 + Python 3.14)")
    # ─────────────────────────────────────────────────────────────────

    t0 = time.time()
    try:
        model = RTDETR(BASE_MODEL)
        model.train(
            data        = str(yaml_path),
            epochs      = EPOCHS,
            imgsz       = IMG_SIZE,
            batch       = BATCH_SIZE,
            device      = device,
            name        = NAMA_TRAINING,
            patience    = 20,        # early stop jika 20 epoch tidak ada perbaikan
            save        = True,
            save_period = 10,        # simpan checkpoint tiap 10 epoch
            # Augmentasi
            hsv_h       = 0.015,
            hsv_s       = 0.7,
            hsv_v       = 0.4,
            fliplr      = 0.5,
            mosaic      = 0.5,
            # Log
            plots       = True,
            verbose     = True,
            workers     = 0,         # hindari multiprocessing crash di Windows Python 3.14
        )
    except KeyboardInterrupt:
        warn("Training dihentikan (Ctrl+C)")
        warn(f"Checkpoint terakhir → runs/detect/{NAMA_TRAINING}/weights/last.pt")
        sys.exit(0)

    elapsed = time.time() - t0
    sep()
    ok(f"Training selesai dalam {int(elapsed//60)}m {int(elapsed%60)}s")

# ═══════════════════════════════════════════════════════════════════
#  LANGKAH 9 — EVALUASI
# ═══════════════════════════════════════════════════════════════════
def run_eval(yaml_path: Path, device):
    hdr("LANGKAH 9 — EVALUASI MODEL (data test)")
    best = Path('runs')/'detect'/NAMA_TRAINING/'weights'/'best.pt'
    if not best.exists():
        warn("best.pt belum ada — jalankan evaluasi manual setelah training selesai:")
        info(f"  yolo val model={best} data={yaml_path} split=test")
        return
    try:
        from ultralytics import RTDETR
        m       = RTDETR(str(best))
        metrics = m.val(data=str(yaml_path), split='test', device=device, verbose=True)
        map50   = metrics.box.map50
        sep()
        log("HASIL mAP50:", 'bold')
        if   map50 >= 0.85: ok(f"mAP50 = {map50:.3f}  →  SANGAT BAIK  — siap deploy ke v.py")
        elif map50 >= 0.70: warn(f"mAP50 = {map50:.3f}  →  BAIK         — pantau hasilnya di lapangan")
        elif map50 >= 0.50: warn(f"mAP50 = {map50:.3f}  →  CUKUP        — tambah data anotasi & ulang training")
        else:               log(f"mAP50 = {map50:.3f}  →  KURANG        — perbaiki kualitas anotasi",'red')
    except Exception as e:
        warn(f"Evaluasi gagal: {e}")

# ═══════════════════════════════════════════════════════════════════
#  LANGKAH 10 — INSTRUKSI DEPLOY KE v.py
# ═══════════════════════════════════════════════════════════════════
def print_deploy(classes: list):
    hdr("LANGKAH 10 — DEPLOY KE v.py")
    best = Path('runs')/'detect'/NAMA_TRAINING/'weights'/'best.pt'
    if best.exists():
        ok(f"Model terbaik → {best.resolve()}")
        print()
        print(c("  Ubah bagian KONFIGURASI di v.py:", 'cyan'))
        print()
        print(c(f"  MODEL_PATH    = r'{best.resolve()}'", 'yellow'))
        print(c(f"  CAR_CLASS_IDS = {{0}}   # satu class: {classes[0] if classes else 'vehicle'}", 'yellow'))
        print()
        log("Garis pembagi zona (LANE_SPLIT_POS) tidak perlu diubah.", 'gray')
        log("Tekan L di v.py untuk kalibrasi visual posisi garis zona.", 'gray')
    else:
        warn("best.pt tidak ditemukan — training mungkin belum selesai")
    sep()

# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    sep()
    log("LABEL STUDIO YOLO ZIP → RT-DETR TRAINING PIPELINE", 'bold')
    log(f"ZIP Input : {ZIP_PATH}", 'cyan')
    log(f"Dataset   : {DATASET_DIR}", 'cyan')
    log(f"Class     : {CLASS_NAMES}", 'cyan')
    log(f"Split     : Train {RATIO_TRAIN*100:.0f}% / "
        f"Valid {RATIO_VALID*100:.0f}% / "
        f"Test {(1-RATIO_TRAIN-RATIO_VALID)*100:.0f}%", 'cyan')
    sep()

    dataset_dir = Path(DATASET_DIR)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # 1. GPU
    device = check_gpu()

    # 2. Ekstrak ZIP
    extract_dir = extract_zip(ZIP_PATH, dataset_dir)

    # 3-4. Baca class dari ZIP
    zip_classes = read_classes(extract_dir)

    # 5. Normalisasi struktur
    normalize_dataset(extract_dir, dataset_dir, zip_classes)

    # 5b. Remap index jika class berbeda urutan
    remap_label_indices(dataset_dir, zip_classes, CLASS_NAMES)

    # 6. Validasi label
    validate_labels(dataset_dir, CLASS_NAMES)

    # 7. Buat data.yaml
    yaml_path = make_yaml(dataset_dir, CLASS_NAMES)

    # 8. Training
    run_training(yaml_path, device)

    # 9. Evaluasi
    run_eval(yaml_path, device)

    # 10. Deploy info
    print_deploy(CLASS_NAMES)


if __name__ == "__main__":
    main()