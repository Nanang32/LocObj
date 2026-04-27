"""
╔══════════════════════════════════════════════════════════════════════╗
║   TRAINING SCRIPT — Roboflow YOLO ZIP → RT-DETR                    ║
║                                                                      ║
║   Input  : file .zip hasil export Roboflow (format YOLOv8)         ║
║   Output : model best.pt siap dipakai di v.py                      ║
║                                                                      ║
║   Cara export di Roboflow:                                          ║
║   Project → Versions → Export Dataset → YOLOv8 → Download ZIP     ║
║                                                                      ║
║   Cara pakai script ini:                                             ║
║   1. Isi ZIP_PATH dan DATASET_DIR di bagian KONFIGURASI            ║
║   2. python train.py                                                 ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os, sys, re, shutil, random, time, zipfile, yaml
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
#  KONFIGURASI — SESUAIKAN SEBELUM MENJALANKAN
# ═══════════════════════════════════════════════════════════════════

# Path ke file ZIP hasil export Roboflow (format YOLOv8)
# Contoh: r"C:\Users\ASUS\Downloads\vehicle-detection.v1.yolov8.zip"
ZIP_PATH      = r"C:\Users\ASUS\Documents\disertasi\LocObj\src\anotation-export\video.mobil.zip"         # ← GANTI

# Folder tujuan ekstrak dan persiapan dataset (dibuat otomatis)
DATASET_DIR   = r"C:\Users\ASUS\Documents\disertasi\LocObj\src\model\dataset\dataset_kendaraan"  # ← GANTI jika perlu

# Nama class — harus sama dengan nama class di Roboflow Anda
# Roboflow akan otomatis dibaca dari data.yaml di dalam ZIP
# Isi ini hanya sebagai fallback jika data.yaml tidak ditemukan
CLASS_NAMES   = ['carback', 'carfront']                                    # ← sesuaikan

# ── Parameter training ────────────────────────
IMG_SIZE      = 640
EPOCHS        = 60
BATCH_SIZE    = 4        # 4GB VRAM→4 | 8GB→8 | 16GB→16
NAMA_TRAINING = "vehicle_roboflow_v1"
BASE_MODEL    = "rtdetr-l.pt"   # model pretrained — didownload otomatis jika belum ada

# ── Split ratio ───────────────────────────────
# Roboflow biasanya sudah split otomatis — variabel ini hanya dipakai
# jika ZIP tidak memiliki struktur train/valid/test yang jelas
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
#  LANGKAH 2 — EKSTRAK ZIP ROBOFLOW
#
#  Roboflow YOLOv8 ZIP memiliki struktur:
#    train/
#      images/  *.jpg
#      labels/  *.txt
#    valid/
#      images/  *.jpg
#      labels/  *.txt
#    test/
#      images/  *.jpg
#      labels/  *.txt
#    data.yaml  ← konfigurasi dataset lengkap (path + class names)
#    README.roboflow.txt
#
#  Berbeda dengan Label Studio, Roboflow MENYERTAKAN gambar di ZIP
#  dan sudah menyertakan data.yaml siap pakai.
# ═══════════════════════════════════════════════════════════════════
def extract_zip(zip_path: str, dataset_dir: Path) -> Path:
    hdr("LANGKAH 2 — EKSTRAK ZIP ROBOFLOW")

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
        # Hitung gambar dan label di dalam ZIP
        imgs = [m for m in members if m.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp'))]
        lbls = [m for m in members if m.endswith('.txt') and 'classes' not in m.lower()]
        info(f"Gambar di ZIP  : {len(imgs)}")
        info(f"Label di ZIP   : {len(lbls)}")
        if len(imgs) == 0:
            err("ZIP tidak mengandung gambar!\n"
                "     Pastikan Anda memilih format YOLOv8 saat export di Roboflow,\n"
                "     bukan format lain seperti COCO JSON atau Pascal VOC.")
        zf.extractall(extract_dir)

    ok("Ekstrak selesai")
    return extract_dir

# ═══════════════════════════════════════════════════════════════════
#  LANGKAH 3 — BACA data.yaml DARI ROBOFLOW
# ═══════════════════════════════════════════════════════════════════
def read_roboflow_yaml(extract_dir: Path) -> tuple:
    """
    Baca data.yaml yang disertakan Roboflow di dalam ZIP.
    Returns: (classes_list, yaml_data_dict)
    Roboflow sudah menyertakan path train/valid/test dan class names.
    """
    yaml_candidates = list(extract_dir.rglob('data.yaml'))
    if not yaml_candidates:
        warn("data.yaml tidak ditemukan di ZIP — akan dibuat dari CLASS_NAMES")
        return CLASS_NAMES, None

    yaml_path = yaml_candidates[0]
    ok(f"data.yaml ditemukan : {yaml_path.relative_to(extract_dir)}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    classes = data.get('names', CLASS_NAMES)
    nc      = data.get('nc', len(classes))
    ok(f"Class dari Roboflow : {classes}  (nc={nc})")

    # Validasi terhadap CLASS_NAMES di konfigurasi
    if sorted(classes) != sorted(CLASS_NAMES):
        warn(f"Class di ZIP    : {classes}")
        warn(f"CLASS_NAMES     : {CLASS_NAMES}")
        warn("Menggunakan class dari ZIP Roboflow")

    return classes, data

# ═══════════════════════════════════════════════════════════════════
#  LANGKAH 4 — BACA classes.txt (fallback)
# ═══════════════════════════════════════════════════════════════════
def read_classes(extract_dir: Path) -> list:
    """Fallback jika data.yaml tidak ada."""
    candidates = list(extract_dir.rglob('classes.txt'))
    if not candidates:
        warn(f"classes.txt tidak ditemukan — pakai CLASS_NAMES: {CLASS_NAMES}")
        return CLASS_NAMES
    lines = [l.strip() for l in candidates[0].read_text(encoding='utf-8').splitlines() if l.strip()]
    ok(f"Class dari classes.txt : {lines}")
    return lines if lines else CLASS_NAMES

# ═══════════════════════════════════════════════════════════════════
#  LANGKAH 5 — NORMALISASI DATASET KE STRUKTUR YOLOV8
#
#  Roboflow sudah pakai struktur YOLOv8 — fungsi ini hanya
#  memastikan folder train/valid/test ada dan merapikan path.
#  Jika struktur sudah benar, data langsung dipakai tanpa copy.
# ═══════════════════════════════════════════════════════════════════
def collect_pairs(extract_dir: Path) -> list:
    img_exts = {'.jpg','.jpeg','.png','.bmp','.webp'}
    pairs    = []
    for ext in img_exts:
        for img in extract_dir.rglob(f'*{ext}'):
            lbl = img.with_suffix('.txt')
            if not lbl.exists():
                parts = list(img.parts)
                for i, part in enumerate(parts):
                    if part.lower() in ('images','image','imgs'):
                        parts[i] = 'labels'
                        lbl = Path(*parts).with_suffix('.txt')
                        break
            if lbl.exists():
                pairs.append((img, lbl))
            else:
                warn(f"Label tidak ada: {img.name} — dilewati")
    return pairs


def normalize_dataset(extract_dir: Path, dataset_dir: Path, classes: list) -> Path:
    hdr("LANGKAH 5 — NORMALISASI STRUKTUR DATASET")

    # ── Cek apakah Roboflow sudah split train/valid/test ─────────
    has_train = (extract_dir/'train'/'images').exists()
    has_valid = (extract_dir/'valid'/'images').exists()
    has_test  = (extract_dir/'test'/'images').exists()

    if has_train and has_valid:
        ok("Struktur Roboflow terdeteksi: train/valid/test sudah ada")
        splits_src = {
            'train': extract_dir/'train',
            'valid': extract_dir/'valid',
        }
        if has_test:
            splits_src['test'] = extract_dir/'test'
        else:
            # Jika tidak ada folder test, bagi valid 50/50
            warn("Folder test/ tidak ada — membagi valid menjadi valid+test (50/50)")

        total = 0
        for split_name, src_dir in splits_src.items():
            out_img = dataset_dir/split_name/'images'
            out_lbl = dataset_dir/split_name/'labels'
            out_img.mkdir(parents=True, exist_ok=True)
            out_lbl.mkdir(parents=True, exist_ok=True)
            src_img = src_dir/'images'
            src_lbl = src_dir/'labels'
            count = 0
            if src_img.exists():
                for img in src_img.iterdir():
                    if img.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}:
                        shutil.copy2(img, out_img/img.name)
                        lbl = src_lbl/(img.stem+'.txt') if src_lbl.exists() else None
                        if lbl and lbl.exists():
                            shutil.copy2(lbl, out_lbl/(img.stem+'.txt'))
                        count += 1
            total += count
            info(f"  {split_name:5s}: {count:4d} gambar")

        # Buat folder test jika belum ada dengan split dari valid
        if not has_test:
            valid_imgs = list((dataset_dir/'valid'/'images').glob('*.*'))
            random.seed(42); random.shuffle(valid_imgs)
            mid = len(valid_imgs)//2
            test_img_dir = dataset_dir/'test'/'images'
            test_lbl_dir = dataset_dir/'test'/'labels'
            test_img_dir.mkdir(parents=True, exist_ok=True)
            test_lbl_dir.mkdir(parents=True, exist_ok=True)
            for img in valid_imgs[mid:]:
                shutil.move(str(img), test_img_dir/img.name)
                lbl = dataset_dir/'valid'/'labels'/(img.stem+'.txt')
                if lbl.exists():
                    shutil.move(str(lbl), test_lbl_dir/(img.stem+'.txt'))
            info(f"  test : {len(valid_imgs)-mid:4d} gambar (dipindah dari valid)")
            info(f"  valid: {mid:4d} gambar (setelah split)")

    else:
        # Fallback: koleksi semua gambar+label lalu split manual
        warn("Struktur Roboflow tidak terdeteksi — split manual 80/10/10")
        all_pairs = collect_pairs(extract_dir)
        if not all_pairs:
            err("Tidak ada pasangan gambar+label ditemukan di ZIP!\n"
                "     Pastikan Anda mengekspor format YOLOv8 dari Roboflow.")
        random.seed(42); random.shuffle(all_pairs)
        n  = len(all_pairs)
        nt = max(1, int(n * RATIO_TRAIN))
        nv = max(1, int(n * RATIO_VALID))
        splits = {'train': all_pairs[:nt], 'valid': all_pairs[nt:nt+nv], 'test': all_pairs[nt+nv:]}
        total  = n
        for split_name, pairs in splits.items():
            out_img = dataset_dir/split_name/'images'
            out_lbl = dataset_dir/split_name/'labels'
            out_img.mkdir(parents=True, exist_ok=True)
            out_lbl.mkdir(parents=True, exist_ok=True)
            for img, lbl in pairs:
                shutil.copy2(img, out_img/img.name)
                shutil.copy2(lbl, out_lbl/(img.stem+'.txt'))
            info(f"  {split_name:5s}: {len(pairs):4d} gambar ({len(pairs)/max(total,1)*100:.0f}%)")

    if total < 50:
        warn(f"Hanya {total} gambar — lanjutkan anotasi di Roboflow untuk hasil lebih baik")
        warn("Direkomendasikan minimal 1.000 gambar untuk mencegah overfitting")

    ok("Normalisasi selesai")
    return dataset_dir


def remap_label_indices(dataset_dir: Path, zip_classes: list, target_classes: list):
    if zip_classes == target_classes:
        return
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
            warn(f"  Class '{name}' tidak ada di ZIP")
    for lbl_file in dataset_dir.rglob('*.txt'):
        if lbl_file.name in ('classes.txt','data.yaml'): continue
        lines = lbl_file.read_text(encoding='utf-8').splitlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            new = mapping.get(int(parts[0]))
            if new is None: continue
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
        warn(f"Ditemukan {errors} baris bermasalah — periksa anotasi di Roboflow")

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
        print(c(f"  CAR_CLASS_IDS = {{0, 1}}   # 0={classes[0] if len(classes)>0 else 'carback'}, 1={classes[1] if len(classes)>1 else 'carfront'}", 'yellow'))
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
    log("ROBOFLOW YOLO ZIP → RT-DETR TRAINING PIPELINE", 'bold')
    log(f"ZIP Input  : {ZIP_PATH}", 'cyan')
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

    # 2. Ekstrak ZIP Roboflow
    extract_dir = extract_zip(ZIP_PATH, dataset_dir)

    # 3. Baca data.yaml + class dari Roboflow ZIP
    zip_classes, rf_yaml = read_roboflow_yaml(extract_dir)

    # 4. Fallback baca classes.txt jika data.yaml tidak ada
    if not zip_classes:
        zip_classes = read_classes(extract_dir)

    # 5. Normalisasi/copy dataset ke DATASET_DIR
    normalize_dataset(extract_dir, dataset_dir, zip_classes)

    # 5b. Remap index class jika urutan berbeda
    remap_label_indices(dataset_dir, zip_classes, CLASS_NAMES)

    # 6. Validasi label
    validate_labels(dataset_dir, CLASS_NAMES)

    # 7. Buat data.yaml final di DATASET_DIR
    yaml_path = make_yaml(dataset_dir, CLASS_NAMES)

    # 8. Training
    run_training(yaml_path, device)

    # 9. Evaluasi
    run_eval(yaml_path, device)

    # 10. Deploy info
    print_deploy(CLASS_NAMES)


if __name__ == "__main__":
    main()