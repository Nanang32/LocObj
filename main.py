"""
╔══════════════════════════════════════════════════════════════════════╗
║   RT-DETR  Vehicle Tracker  —  Custom Model (best.pt)              ║
║                                                                      ║
║   ARSITEKTUR PIPELINE:                                              ║
║                                                                      ║
║   Kamera                                                            ║
║     ↓                                                               ║
║   best.pt  ←── model RT-DETR yang sudah di-fine-tune               ║
║   (berisi arsitektur rtdetr-l + pengetahuan dataset carback/front)  ║
║     ↓                                                               ║
║   Filter lebar bbox  (1600–1850 mm)                                 ║
║   Objek di luar range → diabaikan sepenuhnya                        ║
║     ↓                                                               ║
║   Tracker ID  (ID 1 = objek pertama/terlama, ID 2, 3, ...)         ║
║   ID 1 di-lock dan dipertahankan selama GRACE_SEC detik            ║
║     ↓                                                               ║
║   CSV Logger  (setiap event: MASUK, HILANG, KEMBALI, EXPIRE)       ║
║                                                                      ║
║   CLASS MODEL:                                                       ║
║     0 = carback  (body belakang kendaraan)                          ║
║     1 = carfront (body depan kendaraan)                             ║
║                                                                      ║
║   TOMBOL:                                                            ║
║     Q = keluar                                                       ║
║     R = reset semua track                                            ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import cv2
import csv
import numpy as np
import time
import os
from pathlib import Path
from collections import OrderedDict
from ultralytics import RTDETR

# ══════════════════════════════════════════════════════════════════════
#  KONFIGURASI
# ══════════════════════════════════════════════════════════════════════

# ── Model ─────────────────────────────────────────────────────────────
#
#  best.pt adalah model RT-DETR yang sudah di-fine-tune dengan dataset
#  kendaraan Anda sendiri (carback + carfront). Model ini menggantikan
#  rtdetr-l.pt + filter COCO class secara langsung — tidak perlu dua model.
#
#  Taruh best.pt satu folder dengan main.py, ATAU isi path lengkap:
#  MODEL_PATH = r"C:\Users\ASUS\Documents\disertasi\LocObj\runs\detect\vehicle_roboflow_v1\weights\best.pt"
#
MODEL_PATH =r"C:\Users\ASUS\Documents\disertasi\LocObj\src\model\weight-v2\best.pt"

# ── Class dari model custom (sesuai urutan di Roboflow) ───────────────
#  0 = carback  → body belakang kendaraan
#  1 = carfront → body depan kendaraan
#  Kedua class dideteksi — pembedaan zona (mana depan mana belakang)
#  dilakukan secara geometri berdasarkan posisi bbox di frame.
CLASS_IDS = {0, 1}

# ── Confidence threshold ──────────────────────────────────────────────
#  Model custom bisa lebih rendah dari COCO (0.45) karena sudah
#  spesifik — hanya mengenal kendaraan yang dianotasi.
CONF_THRESHOLD = 0.35

# ── Ukuran kendaraan valid per class (mm) ────────────────────────────
#
#  carback (class 0) = body belakang, kendaraan DEKAT kamera
#    Lebar nyata kendaraan roda 4: 1600–1850mm
#
#  carfront (class 1) = body depan, kendaraan dari arah BERLAWANAN
#    Karena lebih jauh dari kamera, bbox dalam pixel lebih kecil.
#    Konversi mm menghasilkan angka lebih besar → range berbeda.
#    Sesuaikan WIDTH_FRONT_MIN/MAX dengan kondisi lapangan Anda.
#
WIDTH_BACK_MIN_MM  = 3000.0   # carback minimum (mm) — body belakang, ID 1 terkunci
WIDTH_BACK_MAX_MM  = 9999.0   # carback maximum (mm) — tidak ada batas atas (semua jarak)
WIDTH_FRONT_MIN_MM = 4000.0   # carfront minimum (mm) — jalur berlawanan
WIDTH_FRONT_MAX_MM = 5000.0   # carfront maximum (mm) — jalur berlawanan

# ── Konversi pixel ke mm ──────────────────────────────────────────────
#  Satu nilai PIXEL_PER_MM dipakai untuk kedua class.
#  Perbedaan range sudah menangani perbedaan jarak kamera.
#  Cara kalibrasi: ukur lebar kendaraan nyata (mm) ÷ lebar bbox (pixel)
PIXEL_PER_MM = 0.20

# ── Alias untuk backward compatibility ───────────────────────────────
WIDTH_MIN_MM = WIDTH_BACK_MIN_MM
WIDTH_MAX_MM = WIDTH_BACK_MAX_MM

# ── Grace period ──────────────────────────────────────────────────────
#  Berapa detik bbox dipertahankan setelah objek hilang dari frame.
#  Setelah melebihi ini → objek di-expire dan dihapus dari tracking.
GRACE_SEC = 10.0

# ── Parameter tracking ────────────────────────────────────────────────
W_IOU     = 0.50   # bobot IoU antar bbox
W_DIST    = 0.30   # bobot jarak centroid
W_APP     = 0.20   # bobot kemiripan warna/appearance
MIN_MATCH = 0.30   # skor minimum untuk matching track
MAX_DIST  = 400    # jarak pixel maksimum untuk matching

# ══════════════════════════════════════════════════════════════════════
#  WARNA TAMPILAN
# ══════════════════════════════════════════════════════════════════════
#  ID 1 (objek terlama/terkunci) → cyan
#  ID 2, 3, ... → warna bergilir
COLORS = [
    (0, 220, 255),   # ID 1 — cyan
    (0, 255, 120),   # ID 2 — hijau
    (255, 180, 0),   # ID 3 — kuning
    (200, 80, 255),  # ID 4 — ungu
    (255, 80, 80),   # ID 5 — merah
    (80, 180, 255),  # ID 6 — biru
]

# ══════════════════════════════════════════════════════════════════════
#  TERMINAL HELPER
# ══════════════════════════════════════════════════════════════════════
A = {
    'rst': '\033[0m', 'b': '\033[1m',  'cy': '\033[96m',
    'gr':  '\033[92m','yw': '\033[93m','rd':  '\033[91m',
    'gy':  '\033[90m','wh': '\033[97m',
}
def cp(t, *s): print(''.join(A.get(x, '') for x in s) + t + A['rst'])
def sep():     cp("─" * 66, 'gy')
def ts():      return time.strftime('%H:%M:%S')
def fmt(sec):
    if sec < 60: return f"{sec:06.3f}s"
    m = int(sec // 60); return f"{m}:{sec - m*60:06.3f}s"

# ══════════════════════════════════════════════════════════════════════
#  VALIDASI LEBAR BBOX
# ══════════════════════════════════════════════════════════════════════
def bbox_width_mm(bbox):
    """Hitung lebar bbox dalam mm menggunakan konversi pixel."""
    return (bbox[2] - bbox[0]) / PIXEL_PER_MM

def is_valid_width(bbox, cls_id=None):
    """
    Validasi lebar bbox sesuai class:
      carback  (cls_id=0): 1600–1850mm  (kendaraan dekat, body belakang)
      carfront (cls_id=1): 4000–5000mm  (kendaraan jauh, body depan, jalur berlawanan)
    """
    w = bbox_width_mm(bbox)
    if cls_id == 1:   # carfront — jalur berlawanan, lebih jauh
        return WIDTH_FRONT_MIN_MM <= w <= WIDTH_FRONT_MAX_MM, w
    else:             # carback (default) — dekat kamera
        return WIDTH_BACK_MIN_MM  <= w <= WIDTH_BACK_MAX_MM,  w

# ══════════════════════════════════════════════════════════════════════
#  UTILITAS TRACKING
# ══════════════════════════════════════════════════════════════════════
def iou(b1, b2):
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter / union if union > 0 else 0.0

def centroid(b):
    return ((b[0]+b[2])//2, (b[1]+b[3])//2)

def ndist(b1, b2):
    c1 = centroid(b1); c2 = centroid(b2)
    return min(np.hypot(c1[0]-c2[0], c1[1]-c2[1]) / MAX_DIST, 1.0)

def get_app(frame, bbox):
    x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return np.zeros(48)
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h    = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s    = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v    = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    hist = np.concatenate([h, s, v]).astype(np.float32)
    n    = hist.sum()
    return hist / n if n > 0 else hist

def app_sim(h1, h2):
    d = np.dot(h1, h2)
    n = np.linalg.norm(h1) * np.linalg.norm(h2)
    return float(d / n) if n > 0 else 0.0

def mscore(track, bbox, app):
    return (W_IOU * iou(track['bbox'], bbox)
          + W_DIST * (1 - ndist(track['bbox'], bbox))
          + W_APP  * app_sim(track['appearance'], app))

# ══════════════════════════════════════════════════════════════════════
#  CSV LOGGER
#
#  Kolom yang dicatat:
#    timestamp          : waktu event (HH:MM:SS)
#    waktu_unix         : epoch float (presisi tinggi)
#    display_id         : ID tampilan saat event
#    internal_id        : ID internal tracker (stabil sepanjang hidup)
#    class_label        : carback atau carfront
#    event              : MASUK | LOCK_SELESAI | HILANG | KEMBALI |
#                         PROMOSI | EXPIRE
#    durasi_lock_detik  : durasi ID 1 terkunci (diisi saat LOCK_SELESAI)
#    durasi_deteksi_detik: total on-frame (diisi saat EXPIRE)
#    durasi_off_detik   : total off-frame (diisi saat EXPIRE & KEMBALI)
#    lebar_mm           : lebar bbox saat event
#    conf               : confidence score
#    keterangan         : deskripsi event
# ══════════════════════════════════════════════════════════════════════
CLASS_LABEL = {0: 'carback', 1: 'carfront', None: 'unknown'}

class CSVLogger:
    FIELDNAMES = [
        'timestamp', 'waktu_unix', 'display_id', 'internal_id',
        'class_label', 'event', 'durasi_lock_detik',
        'durasi_deteksi_detik', 'durasi_off_detik',
        'lebar_mm', 'conf', 'keterangan',
    ]

    def __init__(self, path='tracking_log.csv'):
        self.path = path
        with open(self.path, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=self.FIELDNAMES).writeheader()
        cp(f"  CSV Logger → {os.path.abspath(self.path)}", 'gr', 'b')
        self._lock_start = {}   # iid → waktu mulai lock (jadi ID 1)

    def _write(self, row: dict):
        full = {f: '' for f in self.FIELDNAMES}
        full.update(row)
        with open(self.path, 'a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=self.FIELDNAMES).writerow(full)

    def _base(self, now, t, iid):
        return {
            'timestamp':   time.strftime('%H:%M:%S', time.localtime(now)),
            'waktu_unix':  f"{now:.3f}",
            'display_id':  t['display_id'],
            'internal_id': iid,
            'class_label': CLASS_LABEL.get(t.get('cls_id'), 'unknown'),
            'lebar_mm':    f"{t['width_mm']:.1f}",
            'conf':        f"{t['conf']:.3f}",
        }

    def log_masuk(self, iid, t, now):
        r = self._base(now, t, iid)
        r['event']      = 'MASUK'
        r['keterangan'] = f"Objek {CLASS_LABEL.get(t.get('cls_id'),'?')} masuk frame"
        self._write(r)
        if t['display_id'] == 1:
            self._lock_start[iid] = now

    def log_promosi(self, iid, old_id, new_id, t, now):
        # Tutup lock lama jika keluar dari ID 1
        if old_id == 1 and iid in self._lock_start:
            dur = now - self._lock_start.pop(iid)
            r   = self._base(now, t, iid)
            r.update({'event': 'LOCK_SELESAI',
                      'durasi_lock_detik': f"{dur:.3f}",
                      'keterangan': f"Lock ID 1 selesai ({dur:.2f}s) → naik ke ID {new_id}"})
            self._write(r)
        # Mulai lock baru jika masuk ke ID 1
        if new_id == 1:
            self._lock_start[iid] = now
        r = self._base(now, t, iid)
        r.update({'event': 'PROMOSI',
                  'keterangan': f"ID {old_id} → ID {new_id}"})
        self._write(r)

    def log_hilang(self, iid, t, now):
        r = self._base(now, t, iid)
        r['event']      = 'HILANG'
        r['keterangan'] = f"Hilang dari frame — grace {GRACE_SEC:.0f}s"
        self._write(r)

    def log_kembali(self, iid, t, miss_dur, now):
        r = self._base(now, t, iid)
        r.update({'event': 'KEMBALI',
                  'durasi_off_detik': f"{miss_dur:.3f}",
                  'keterangan': f"Kembali setelah {miss_dur:.2f}s off-frame"})
        self._write(r)

    def log_expire(self, iid, exp, now):
        on_dur   = exp.get('last_on', exp['enter_time']) - exp['enter_time']
        miss_dur = now - exp.get('last_seen', now)
        tot_off  = exp.get('total_off', 0.0) + miss_dur
        # Tutup lock jika objek expire saat masih ID 1
        if iid in self._lock_start:
            lock_dur = now - self._lock_start.pop(iid)
            self._write({
                'timestamp':         time.strftime('%H:%M:%S', time.localtime(now)),
                'waktu_unix':        f"{now:.3f}",
                'display_id':        exp['display_id'],
                'internal_id':       iid,
                'class_label':       CLASS_LABEL.get(exp.get('cls_id'), 'unknown'),
                'event':             'LOCK_SELESAI',
                'durasi_lock_detik': f"{lock_dur:.3f}",
                'lebar_mm':          f"{exp['width_mm']:.1f}",
                'keterangan':        'Lock ditutup — objek expire',
            })
        self._write({
            'timestamp':             time.strftime('%H:%M:%S', time.localtime(now)),
            'waktu_unix':            f"{now:.3f}",
            'display_id':            exp['display_id'],
            'internal_id':           iid,
            'class_label':           CLASS_LABEL.get(exp.get('cls_id'), 'unknown'),
            'event':                 'EXPIRE',
            'durasi_deteksi_detik':  f"{on_dur:.3f}",
            'durasi_off_detik':      f"{tot_off:.3f}",
            'lebar_mm':              f"{exp['width_mm']:.1f}",
            'keterangan':            (f"on {on_dur:.2f}s | off {tot_off:.2f}s | "
                                      f"hilang {miss_dur:.1f}s"),
        })

    def flush_active(self, tracks, now):
        """Tutup semua lock aktif saat program berhenti."""
        for iid, lock_t in list(self._lock_start.items()):
            t = tracks.get(iid)
            if not t: continue
            lock_dur = now - lock_t
            on_dur   = (now - t['enter_time']) - t.get('total_off', 0.0)
            self._write({
                'timestamp':             time.strftime('%H:%M:%S', time.localtime(now)),
                'waktu_unix':            f"{now:.3f}",
                'display_id':            t['display_id'],
                'internal_id':           iid,
                'class_label':           CLASS_LABEL.get(t.get('cls_id'), 'unknown'),
                'event':                 'LOCK_SELESAI',
                'durasi_lock_detik':     f"{lock_dur:.3f}",
                'durasi_deteksi_detik':  f"{on_dur:.3f}",
                'lebar_mm':              f"{t['width_mm']:.1f}",
                'keterangan':            'Program berhenti — lock ditutup paksa',
            })
        cp(f"  CSV disimpan → {os.path.abspath(self.path)}", 'gr', 'b')

# ══════════════════════════════════════════════════════════════════════
#  ID MANAGER — tracker multi-objek, satu antrian ID global
#
#  Aturan konsistensi ID:
#
#  SATU ANTRIAN GLOBAL berdasarkan waktu masuk (enter_time):
#    ID 1 = objek PERTAMA yang masuk frame → di-LOCK sebagai referensi
#    ID 2 = objek kedua (carback ATAU carfront)
#    ID 3 = objek ketiga (carback ATAU carfront)
#    dst...
#
#  Contoh:
#    t=0s  carback masuk  → ID 1 [LOCK]
#    t=2s  carfront masuk → ID 2
#    t=3s  carback masuk  → ID 3
#    t=5s  carfront masuk → ID 4
#
#  ID 1 TIDAK berubah meski objek baru datang dari class apapun.
#  Jika ID 1 expire → ID 2 naik jadi ID 1 (promosi otomatis).
#
#  Matching deteksi tetap dibatasi per class (carback hanya
#  cocok ke track carback, carfront ke track carfront) agar
#  tidak terjadi ID tertukar antar jalur.
# ══════════════════════════════════════════════════════════════════════
class IDManager:
    def __init__(self):
        self.tracks = OrderedDict()   # semua track (carback + carfront)
        self._ctr   = 0               # counter ID internal unik

    def _rank(self):
        """
        Satu antrian ID global berdasarkan waktu masuk frame (enter_time).
        Carback dan carfront berbaur dalam satu urutan:
          - Siapa masuk pertama dapat ID 1, kedua dapat ID 2, dst.
          - ID 1 selalu dikunci (carback atau carfront apapun yang masuk duluan)
          - Tidak ada antrian terpisah per class
        """
        semua = sorted(self.tracks.items(), key=lambda x: x[1]['enter_time'])
        for rank, (_, t) in enumerate(semua, start=1):
            t['display_id'] = rank

    def _update_pool(self, frame, detections, now):
        """
        Jalankan greedy matching antara detections dan self.tracks,
        kembalikan daftar track yang tidak terpakai (baru).
        """
        used, matched = {}, {}
        for tid, track in self.tracks.items():
            best_score, best_idx = -1, -1
            for i, d in enumerate(detections):
                if i in used: continue
                # Hanya cocokkan dengan track ber-class sama
                if d.get('cls_id') != track.get('cls_id'): continue
                score = mscore(track, d['bbox'], d['appearance'])
                if score > best_score:
                    best_score, best_idx = score, i
            if best_score >= MIN_MATCH:
                matched[tid] = best_idx
                used[best_idx] = True

        # Update track yang cocok
        for tid, di in matched.items():
            d = detections[di]; t = self.tracks[tid]
            t['bbox']       = d['bbox']
            t['conf']       = d['conf']
            t['width_mm']   = d['width_mm']
            t['cls_id']     = d['cls_id']
            t['appearance'] = d['appearance']
            t['last_seen']  = now
            t['missing']    = False
            t['last_on']    = now

        # Tandai track yang tidak cocok sebagai hilang
        for tid in self.tracks:
            if tid not in matched:
                self.tracks[tid]['missing'] = True

        # Kembalikan index deteksi yang belum punya track
        return [i for i in range(len(detections)) if i not in used]

    def update(self, frame, detections):
        now = time.time()
        for d in detections:
            d['appearance'] = get_app(frame, d['bbox'])

        # ── Matching greedy per class (carback tidak akan cocok ke carfront)
        new_idxs = self._update_pool(frame, detections, now)

        # ── Daftarkan deteksi baru yang tidak cocok
        for i in new_idxs:
            d = detections[i]
            self.tracks[self._ctr] = {
                'bbox':       d['bbox'],
                'conf':       d['conf'],
                'width_mm':   d['width_mm'],
                'cls_id':     d['cls_id'],
                'appearance': d['appearance'],
                'enter_time': now,
                'last_seen':  now,
                'last_on':    now,
                'off_time':   None,
                'total_off':  0.0,
                'missing':    False,
                'display_id': None,
            }
            self._ctr += 1

        # ── Hitung durasi off-frame per track
        for tid, t in self.tracks.items():
            if t['missing']:
                if t.get('off_time') is None:
                    t['off_time'] = now
            else:
                if t.get('off_time') is not None:
                    t['total_off'] += now - t['off_time']
                    t['off_time']   = None

        # ── Rank ID per class secara terpisah
        self._rank()

        # ── Expire track yang hilang lebih dari GRACE_SEC
        expired = [(k, dict(t)) for k, t in self.tracks.items()
                   if t['missing'] and (now - t['last_seen']) > GRACE_SEC]
        for tid, _ in expired:
            del self.tracks[tid]

        # ── Re-rank setelah expire (ID otomatis naik jika ID 1 expire)
        self._rank()
        return self.tracks, expired

# ══════════════════════════════════════════════════════════════════════
#  EVENT LOGGER — terminal + CSV
# ══════════════════════════════════════════════════════════════════════
class EventLogger:
    def __init__(self, csv_logger: CSVLogger):
        self.csv         = csv_logger
        self._state      = {}
        self._did_map    = {}
        self._miss_start = {}
        self._last_tick  = {}

    def update(self, tracks, expired_log):
        now = time.time()

        # Objek baru masuk
        for iid, t in tracks.items():
            if iid not in self._did_map:
                lbl = CLASS_LABEL.get(t.get('cls_id'), '?')
                sep()
                cp(f"  [{ts()}]  + MASUK    ID {t['display_id']}  [{lbl}]  "
                   f"{t['width_mm']:.0f}mm  conf {t['conf']:.2f}", 'gr', 'b')
                self._did_map[iid] = t['display_id']
                self._state[iid]   = 'active'
                self.csv.log_masuk(iid, t, now)

        # Promosi ID (objek naik/turun antrian)
        for iid, t in tracks.items():
            prev = self._did_map.get(iid)
            if prev is not None and t['display_id'] != prev:
                sep()
                cp(f"  [{ts()}]  ^ PROMOSI  ID {prev} → ID {t['display_id']}  "
                   f"on {fmt(t['last_on'] - t['enter_time'])}", 'cy', 'b')
                self.csv.log_promosi(iid, prev, t['display_id'], t, now)
                self._did_map[iid] = t['display_id']

        # Objek mulai hilang dari frame
        for iid, t in tracks.items():
            if t['missing'] and self._state.get(iid) == 'active':
                self._state[iid]      = 'missing'
                self._miss_start[iid] = now
                self._last_tick[iid]  = now
                sep()
                cp(f"  [{ts()}]  o HILANG   ID {t['display_id']}  "
                   f"| grace {GRACE_SEC:.0f}s", 'yw', 'b')
                self.csv.log_hilang(iid, t, now)

        # Countdown tiap detik untuk objek yang sedang hilang
        for iid, t in tracks.items():
            if self._state.get(iid) == 'missing':
                miss_s = now - self._miss_start.get(iid, now)
                if now - self._last_tick.get(iid, now) >= 1.0:
                    remain = max(0.0, GRACE_SEC - miss_s)
                    cp(f"         | ID {t['display_id']}  off {miss_s:.1f}s  "
                       f"sisa {remain:.1f}s", 'yw')
                    self._last_tick[iid] = now

        # Objek kembali terdeteksi
        for iid, t in tracks.items():
            if not t['missing'] and self._state.get(iid) == 'missing':
                miss_s           = now - self._miss_start.get(iid, now)
                self._state[iid] = 'active'
                sep()
                cp(f"  [{ts()}]  < KEMBALI  ID {t['display_id']}  "
                   f"off {fmt(miss_s)}", 'cy', 'b')
                self.csv.log_kembali(iid, t, miss_s, now)

        # Objek expire (hilang > GRACE_SEC)
        for iid, exp in expired_log:
            miss_s   = now - exp.get('last_seen', now)
            on_dur   = exp.get('last_on', exp['enter_time']) - exp['enter_time']
            tot_off  = exp.get('total_off', 0.0) + miss_s
            lbl      = CLASS_LABEL.get(exp.get('cls_id'), '?')
            sep()
            cp(f"  [{ts()}]  x EXPIRE   ID {exp['display_id']}  [{lbl}]  "
               f"on {fmt(on_dur)}  off {fmt(tot_off)}", 'rd', 'b')
            self.csv.log_expire(iid, exp, now)

            # Tampilkan siapa ID 1 berikutnya
            next_id1 = next((t for t in tracks.values() if t['display_id'] == 1), None)
            if next_id1:
                cp(f"         └ ID 1 baru: masuk {time.strftime('%H:%M:%S', time.localtime(next_id1['enter_time']))}  "
                   f"aktif {fmt(now - next_id1['enter_time'])}  "
                   f"{next_id1['width_mm']:.0f}mm", 'b', 'cy')
            else:
                cp("         └ Tidak ada objek aktif.", 'gy')

            if tracks:
                antrian = "  ".join(
                    f"ID{t['display_id']}({fmt(now - t['enter_time'])})"
                    for t in sorted(tracks.values(), key=lambda x: x['display_id'])
                )
                cp(f"         Antrian: {antrian}", 'gy')
            sep()

        # Cleanup state untuk objek yang sudah tidak aktif
        alive = set(tracks.keys())
        for d in [self._state, self._did_map, self._miss_start, self._last_tick]:
            for k in [k for k in list(d) if k not in alive]:
                del d[k]

# ══════════════════════════════════════════════════════════════════════
#  ANOTASI FRAME — gambar bbox, ID, label, info
# ══════════════════════════════════════════════════════════════════════
def draw_tracks(frame, tracks):
    now = time.time()
    for _, t in tracks.items():
        did    = t['display_id']
        color  = COLORS[(did - 1) % len(COLORS)]
        x1, y1, x2, y2 = [int(v) for v in t['bbox']]
        miss   = t.get('missing', False)
        miss_s = (now - t['last_seen']) if miss else 0.0
        thick  = 3 if did == 1 else 2

        tot_off = t.get('total_off', 0.0)
        if miss and t.get('off_time'):
            tot_off += now - t['off_time']
        on_dur    = (now - t['enter_time']) - tot_off
        enter_str = time.strftime('%H:%M:%S', time.localtime(t['enter_time']))
        lbl       = CLASS_LABEL.get(t.get('cls_id'), '?')

        # ── Bbox: solid jika terdeteksi, putus-putus jika hilang (grace)
        if miss:
            dash, gap = 12, 6
            for p1, p2 in [((x1,y1),(x2,y1)), ((x2,y1),(x2,y2)),
                            ((x2,y2),(x1,y2)), ((x1,y2),(x1,y1))]:
                tot = int(np.hypot(p2[0]-p1[0], p2[1]-p1[1]))
                for s in range(0, tot, dash + gap):
                    ex = int(p1[0] + (p2[0]-p1[0]) * min(s+dash, tot) / tot)
                    ey = int(p1[1] + (p2[1]-p1[1]) * min(s+dash, tot) / tot)
                    sx = int(p1[0] + (p2[0]-p1[0]) * s / tot)
                    sy = int(p1[1] + (p2[1]-p1[1]) * s / tot)
                    cv2.line(frame, (sx, sy), (ex, ey), color, thick)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        # ── Badge ID + class label
        # Satu antrian global: semua pakai "ID N [class]"
        # ID 1 = terkunci (border putih tebal) apapun classnya
        id_txt   = f"ID {did} {lbl}"
        id_scale = 0.75 if did == 1 else 0.62
        is_lock  = (did == 1)
        (iw, ih), _ = cv2.getTextSize(id_txt, cv2.FONT_HERSHEY_DUPLEX, id_scale, 2)
        pad = 6
        cv2.rectangle(frame, (x1, y1), (x1+iw+pad*2, y1+ih+pad*2), color, -1)
        if is_lock:
            cv2.rectangle(frame, (x1, y1), (x1+iw+pad*2, y1+ih+pad*2), (255,255,255), 2)
        cv2.putText(frame, id_txt, (x1+pad, y1+ih+pad-1),
                    cv2.FONT_HERSHEY_DUPLEX, id_scale, (255,255,255), 2, cv2.LINE_AA)

        # ── Penggaris lebar mm
        my = y2 + 10
        cv2.line(frame, (x1, my), (x2, my), color, 1)
        cv2.line(frame, (x1, my-5), (x1, my+5), color, 2)
        cv2.line(frame, (x2, my-5), (x2, my+5), color, 2)
        cv2.putText(frame, f"{t['width_mm']:.0f}mm",
                    ((x1+x2)//2-26, my+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)

        # ── Info rows di atas bbox
        info_rows = [
            (f"Masuk: {enter_str}",      color,        (0,0,0)),
            (f"ON: {fmt(on_dur)}",        (0,180,60),   (255,255,255)),
        ]
        if miss and miss_s < GRACE_SEC:
            remain = max(0.0, GRACE_SEC - miss_s)
            info_rows.append((f"OFF {miss_s:.1f}s sisa {remain:.0f}s",
                               (0,100,210), (255,255,255)))
        line_h = 18
        for i, (row_lbl, bg, tc) in enumerate(reversed(info_rows)):
            (tw, th), _ = cv2.getTextSize(row_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            by = y1 - 6 - i * (line_h + 2)
            if by < th + 6:
                by = y2 + 36 + i * (line_h + 2)
            cv2.rectangle(frame, (x1, by-th-2), (x1+tw+4, by+2), bg, -1)
            cv2.putText(frame, row_lbl, (x1+2, by),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, tc, 1, cv2.LINE_AA)

        # ── Titik centroid
        cx, cy = centroid(t['bbox'])
        cv2.circle(frame, (cx, cy), 4, color, -1)


def draw_hud(frame, tracks, model_name, fps_display=0.0):
    """HUD atas dan bawah frame."""
    now  = time.time()
    fh, fw = frame.shape[:2]

    # Bar atas
    cv2.rectangle(frame, (0, 0), (fw, 36), (15, 15, 15), -1)
    cv2.putText(frame,
                f"Model: {model_name}  |  Aktif: {len(tracks)}  |  "
                f"Back: {WIDTH_BACK_MIN_MM:.0f}-{WIDTH_BACK_MAX_MM:.0f}mm  |  "
                f"Front: {WIDTH_FRONT_MIN_MM:.0f}-{WIDTH_FRONT_MAX_MM:.0f}mm  |  "
                f"Grace: {GRACE_SEC:.0f}s  |  [Q]Keluar [R]Reset",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (200, 200, 200), 1, cv2.LINE_AA)

    # FPS + device info di pojok kanan atas
    fps_color = (0, 255, 100) if fps_display >= 20 else (0, 200, 255) if fps_display >= 10 else (0, 100, 255)
    fps_txt   = f"{fps_display:.1f} FPS"
    (fw_txt, _), _ = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.putText(frame, fps_txt,
                (fw - fw_txt - 10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, fps_color, 2, cv2.LINE_AA)

    # Bar bawah — ringkasan semua track aktif
    if tracks:
        parts = []
        for t in sorted(tracks.values(), key=lambda x: x['display_id']):
            tot = t.get('total_off', 0.0)
            if t.get('missing') and t.get('off_time'): tot += now - t['off_time']
            on   = (now - t['enter_time']) - tot
            lbl_ = CLASS_LABEL.get(t.get('cls_id'), '?')
            lock = " LOCK" if t['display_id'] == 1 else ""
            parts.append(f"ID{t['display_id']}[{lbl_}{lock}] {fmt(on)}")
        cv2.rectangle(frame, (0, fh-28), (fw, fh), (15, 15, 15), -1)
        cv2.putText(frame, "  |  ".join(parts), (8, fh-9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

# ══════════════════════════════════════════════════════════════════════
#  VALIDASI MODEL SAAT STARTUP
# ══════════════════════════════════════════════════════════════════════
def validate_model_path(path: str) -> str:
    """Cek keberadaan file model. Return path absolut yang valid."""
    p = Path(path)
    if not p.exists():
        # Coba cari di folder yang sama dengan script ini
        alt = Path(__file__).parent / p.name
        if alt.exists():
            return str(alt)
        raise FileNotFoundError(
            f"\n  File model tidak ditemukan: {path}\n"
            f"  Pastikan best.pt ada di folder yang sama dengan main.py\n"
            f"  atau isi MODEL_PATH dengan path lengkap."
        )
    return str(p.resolve())

# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    sep()
    cp("  RT-DETR  Vehicle Tracker  |  Custom Model", 'b', 'wh')
    cp(f"  Model      : {MODEL_PATH}", 'cy')
    cp(f"  Class      : 0=carback  1=carfront", 'cy')
    cp(f"  carback : {WIDTH_BACK_MIN_MM:.0f}–{WIDTH_BACK_MAX_MM:.0f} mm  |  "
       f"carfront: {WIDTH_FRONT_MIN_MM:.0f}–{WIDTH_FRONT_MAX_MM:.0f} mm  |  "
       f"Conf: {CONF_THRESHOLD}  |  Grace: {GRACE_SEC:.0f}s", 'cy')
    cp("  [Q] Keluar  |  [R] Reset track", 'gy')
    sep()

    # ── Validasi model sebelum mulai
    try:
        model_path = validate_model_path(MODEL_PATH)
    except FileNotFoundError as e:
        cp(str(e), 'rd'); return

    ok_model_name = Path(model_path).name
    cp(f"  Model dimuat: {model_path}", 'gr', 'b')

    # ── Deteksi dan aktifkan GPU otomatis ────────────────────────────
    import torch
    if torch.cuda.is_available():
        device    = 0   # GPU index 0
        gpu_name  = torch.cuda.get_device_name(0)
        vram_gb   = torch.cuda.get_device_properties(0).total_memory / 1e9
        # Fix cuDNN untuk CUDA 12.8 + Python 3.14
        torch.backends.cudnn.enabled          = False
        torch.backends.cudnn.benchmark        = False
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32   = False
        cp(f"  Device : GPU — {gpu_name}  ({vram_gb:.1f} GB VRAM)", 'gr', 'b')
        cp("  cuDNN  : dinonaktifkan (stabil untuk CUDA 12.8)", 'gy')
    else:
        device = "cpu"
        cp("  Device : CPU — GPU tidak terdeteksi (inferensi lambat)", 'yw')
        cp("  Pastikan PyTorch CUDA terinstall: pip install torch --index-url https://download.pytorch.org/whl/cu124", 'yw')

    # ── Inisialisasi
    # ── Output CSV ke folder src\csv-output\ ──────────────────────────
    csv_dir  = Path(__file__).parent / "src" / "csv-output"
    csv_dir.mkdir(parents=True, exist_ok=True)   # buat folder jika belum ada
    csv_filename = csv_dir / f"tracking_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    csv_logger   = CSVLogger(str(csv_filename))
    model        = RTDETR(model_path)
    model.to(device)   # pindahkan model ke GPU atau CPU
    manager      = IDManager()
    ev_logger    = EventLogger(csv_logger)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cp("  Kamera tidak ditemukan.", 'rd'); return

    # ── Resolusi kamera dikunci ke 640×640 standar RT-DETR ────────────
    # imgsz=640 adalah standar input model RT-DETR — tidak diubah
    # Ini memastikan frame yang masuk ke model konsisten dengan training
    DISP_W = 640
    DISP_H = 640
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISP_H)

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cp(f"  Resolusi kamera : {cam_w}×{cam_h} px  (standar RT-DETR 640)", 'cy')

    WIN = f"Vehicle Tracker — {ok_model_name}"
    try:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, DISP_W, DISP_H)
    except cv2.error:
        try:
            cv2.namedWindow(WIN, 0)
            cv2.resizeWindow(WIN, DISP_W, DISP_H)
        except cv2.error as e:
            cp(f"  OpenCV GUI error: {e}", 'rd')
            cp("  Install: pip uninstall opencv-python-headless -y && pip install opencv-python", 'yw')
            return
    sep()

    # ── FPS counter ───────────────────────────────────────────────────
    fps_counter = 0
    fps_display = 0.0
    fps_timer   = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            fh, fw = frame.shape[:2]

            # Hitung FPS setiap 1 detik
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                fps_display = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer   = time.time()

            # ── Deteksi dengan best.pt
            #    Model hanya mengenal carback(0) dan carfront(1).
            #    Tidak ada class COCO — filter class dilakukan di sini
            #    hanya sebagai lapisan keamanan.
            raw        = list(model(frame, stream=True))
            valid_dets = []

            for r in raw:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])

                    # Lapisan keamanan: hanya terima class yang dikenal
                    if cls_id not in CLASS_IDS or conf < CONF_THRESHOLD:
                        continue

                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    bbox = [x1, y1, x2, y2]

                    # Filter lebar per class:
                    #   carback  (0): 1600–1850mm (dekat, body belakang)
                    #   carfront (1): 4000–5000mm (jauh, jalur berlawanan)
                    valid, w_mm = is_valid_width(bbox, cls_id)
                    if not valid:
                        continue

                    valid_dets.append({
                        'bbox':     bbox,
                        'conf':     conf,
                        'width_mm': w_mm,
                        'cls_id':   cls_id,   # simpan untuk label display
                    })

            # ── Update tracker
            tracks, expired = manager.update(frame, valid_dets)

            # ── Log event ke terminal dan CSV
            ev_logger.update(tracks, expired)

            # ── Gambar hasil ke frame
            draw_tracks(frame, tracks)
            draw_hud(frame, tracks, ok_model_name, fps_display)

            cv2.imshow(WIN, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            if key in (ord('r'), ord('R')):
                csv_logger.flush_active(manager.tracks, time.time())
                manager   = IDManager()
                ev_logger = EventLogger(csv_logger)
                sep(); cp("  Track di-reset.", 'yw'); sep()

    finally:
        csv_logger.flush_active(manager.tracks, time.time())
        cap.release()
        cv2.destroyAllWindows()
        sep()
        cp("  Program selesai.", 'gr')
        sep()


if __name__ == "__main__":
    main()