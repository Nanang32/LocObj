import cv2
import numpy as np
import time
from collections import OrderedDict
from ultralytics import RTDETR

# ─────────────────────────────────────────────
#  KONFIGURASI FIXED — tidak perlu kalibrasi
#  Asumsi: 1 pixel = 1 mm (skala 1:1)
#  Sesuaikan PIXEL_PER_MM jika kamera berbeda
# ─────────────────────────────────────────────
CAR_CLASS_IDS  = {2, 5, 7}     # COCO: car, bus, truck
CONF_THRESHOLD = 0.45

# Range lebar bodi mobil yang DIIZINKAN (mm)
WIDTH_MIN_MM   = 1600.0
WIDTH_MAX_MM   = 1850.0

# ── Konversi pixel ke mm ──────────────────────
# Ukur lebar mobil referensi di layar (px), lalu bagi dengan lebar nyatanya (mm).
# Contoh: mobil 1700mm tampak 340px → PIXEL_PER_MM = 340/1700 = 0.2
# Ubah nilai ini sesuai setup kamera Anda.
PIXEL_PER_MM   = 0.20           # ← SESUAIKAN INI

# Grace period: berapa detik bbox dipertahankan setelah hilang dari frame
GRACE_SEC      = 10.0           # tepat 10 detik → di atas 10 detik bbox hilang

# Tracking
W_IOU     = 0.50
W_DIST    = 0.30
W_APP     = 0.20
MIN_MATCH = 0.30
MAX_DIST  = 400

# ─────────────────────────────────────────────
#  ANSI TERMINAL
# ─────────────────────────────────────────────
A = {'rst':'\033[0m','b':'\033[1m','cy':'\033[96m','gr':'\033[92m',
     'yw':'\033[93m','rd':'\033[91m','gy':'\033[90m','wh':'\033[97m'}
def cp(t,*s): print(''.join(A.get(x,'') for x in s)+t+A['rst'])
def sep(): cp("─"*64,'gy')
def ts(): return time.strftime('%H:%M:%S')
def fmt(sec):
    if sec < 60: return f"{sec:06.3f}s"
    m = int(sec//60); return f"{m}:{sec-m*60:06.3f}s"

# ─────────────────────────────────────────────
#  VALIDASI LEBAR — fixed, tanpa kalibrasi
# ─────────────────────────────────────────────
def bbox_width_mm(bbox):
    """Konversi lebar bbox pixel → mm menggunakan PIXEL_PER_MM."""
    return (bbox[2] - bbox[0]) / PIXEL_PER_MM

def is_valid_width(bbox):
    w = bbox_width_mm(bbox)
    return WIDTH_MIN_MM <= w <= WIDTH_MAX_MM, w

# ─────────────────────────────────────────────
#  UTILITAS TRACKING
# ─────────────────────────────────────────────
def iou(b1, b2):
    ix1=max(b1[0],b2[0]); iy1=max(b1[1],b2[1])
    ix2=min(b1[2],b2[2]); iy2=min(b1[3],b2[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    u=(b1[2]-b1[0])*(b1[3]-b1[1])+(b2[2]-b2[0])*(b2[3]-b2[1])-inter
    return inter/u if u>0 else 0.0

def centroid(b): return ((b[0]+b[2])//2,(b[1]+b[3])//2)

def ndist(b1,b2):
    c1=centroid(b1); c2=centroid(b2)
    return min(np.hypot(c1[0]-c2[0],c1[1]-c2[1])/MAX_DIST,1.0)

def get_app(frame,bbox):
    x1,y1,x2,y2=[max(0,int(v)) for v in bbox]
    roi=frame[y1:y2,x1:x2]
    if roi.size==0: return np.zeros(48)
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    h=cv2.calcHist([hsv],[0],None,[16],[0,180]).flatten()
    s=cv2.calcHist([hsv],[1],None,[16],[0,256]).flatten()
    v=cv2.calcHist([hsv],[2],None,[16],[0,256]).flatten()
    hist=np.concatenate([h,s,v]).astype(np.float32)
    n=hist.sum(); return hist/n if n>0 else hist

def app_sim(h1,h2):
    d=np.dot(h1,h2); n=np.linalg.norm(h1)*np.linalg.norm(h2)
    return float(d/n) if n>0 else 0.0

def mscore(track,bbox,app):
    return (W_IOU*iou(track['bbox'],bbox)
           +W_DIST*(1-ndist(track['bbox'],bbox))
           +W_APP*app_sim(track['appearance'],app))

# ─────────────────────────────────────────────
#  ID MANAGER
# ─────────────────────────────────────────────
class IDManager:
    def __init__(self):
        self.tracks  = OrderedDict()
        self._ctr    = 0

    def _rank(self):
        for rank,(_,t) in enumerate(self.tracks.items(),start=1):
            t['display_id'] = rank

    def update(self, frame, detections):
        now = time.time()
        for d in detections:
            d['appearance'] = get_app(frame, d['bbox'])

        # Greedy matching
        used, matched = {}, {}
        for tid, track in self.tracks.items():
            bs, bi = -1, -1
            for i, d in enumerate(detections):
                if i in used: continue
                s = mscore(track, d['bbox'], d['appearance'])
                if s > bs: bs, bi = s, i
            if bs >= MIN_MATCH:
                matched[tid] = bi; used[bi] = True

        # Update track yang cocok
        for tid, di in matched.items():
            d = detections[di]; t = self.tracks[tid]
            t['bbox']       = d['bbox']
            t['conf']       = d['conf']
            t['width_mm']   = d['width_mm']
            t['appearance'] = d['appearance']
            t['last_seen']  = now
            t['missing']    = False
            # Catat kapan terakhir ON frame (ada di deteksi)
            t['last_on']    = now

        # Tandai hilang
        for tid in self.tracks:
            if tid not in matched:
                self.tracks[tid]['missing'] = True

        # Track baru
        for i, d in enumerate(detections):
            if i not in used:
                self.tracks[self._ctr] = {
                    'bbox':       d['bbox'],
                    'conf':       d['conf'],
                    'width_mm':   d['width_mm'],
                    'appearance': d['appearance'],
                    'enter_time': now,   # pertama kali masuk frame
                    'last_seen':  now,
                    'last_on':    now,   # terakhir aktif on-frame
                    'off_time':   None,  # kapan mulai off-frame
                    'total_off':  0.0,   # akumulasi waktu off-frame
                    'missing':    False,
                    'display_id': None,
                }
                self._ctr += 1

        # Hitung durasi off-frame per track
        for tid, t in self.tracks.items():
            if t['missing']:
                if t.get('off_time') is None:
                    t['off_time'] = now  # baru mulai off
            else:
                if t.get('off_time') is not None:
                    # Baru kembali — akumulasi off duration
                    t['total_off'] += now - t['off_time']
                    t['off_time']  = None

        self._rank()

        # Expire: hilang > GRACE_SEC (10 detik)
        expired = [dict(t) for _, t in self.tracks.items()
                   if t['missing'] and (now - t['last_seen']) > GRACE_SEC]
        for tid in [k for k, t in self.tracks.items()
                    if t['missing'] and (now - t['last_seen']) > GRACE_SEC]:
            del self.tracks[tid]

        self._rank()
        return self.tracks, expired

# ─────────────────────────────────────────────
#  TERMINAL EVENT LOGGER
# ─────────────────────────────────────────────
class EventLogger:
    def __init__(self):
        self._state      = {}
        self._did_map    = {}
        self._miss_start = {}
        self._last_tick  = {}

    def update(self, tracks, expired_log):
        now = time.time()

        # Objek baru masuk
        for iid, t in tracks.items():
            if iid not in self._did_map:
                sep()
                cp(f"  [{ts()}]  + MASUK    ID {t['display_id']}  "
                   f"lebar {t['width_mm']:.1f}mm  conf {t['conf']:.2f}", 'gr','b')
                self._did_map[iid] = t['display_id']
                self._state[iid]   = 'active'

        # Promosi ID
        for iid, t in tracks.items():
            prev = self._did_map.get(iid)
            if prev is not None and t['display_id'] != prev:
                sep()
                cp(f"  [{ts()}]  ^ PROMOSI  ID {prev} -> ID {t['display_id']}  "
                   f"on-frame {fmt(t['last_on']-t['enter_time'])}", 'cy','b')
                self._did_map[iid] = t['display_id']

        # Mulai hilang
        for iid, t in tracks.items():
            if t['missing'] and self._state.get(iid) == 'active':
                self._state[iid]      = 'missing'
                self._miss_start[iid] = now
                self._last_tick[iid]  = now
                sep()
                cp(f"  [{ts()}]  o HILANG   ID {t['display_id']}  "
                   f"| bbox hilang maks {GRACE_SEC:.0f}s", 'yw','b')

        # Countdown tiap 1 detik
        for iid, t in tracks.items():
            if self._state.get(iid) == 'missing':
                miss_s = now - self._miss_start.get(iid, now)
                if now - self._last_tick.get(iid, now) >= 1.0:
                    remain = max(0.0, GRACE_SEC - miss_s)
                    cp(f"         | ID {t['display_id']}  off {miss_s:.1f}s  "
                       f"sisa grace {remain:.1f}s", 'yw')
                    self._last_tick[iid] = now

        # Kembali terdeteksi
        for iid, t in tracks.items():
            if not t['missing'] and self._state.get(iid) == 'missing':
                miss_s = now - self._miss_start.get(iid, now)
                self._state[iid] = 'active'
                sep()
                cp(f"  [{ts()}]  < KEMBALI  ID {t['display_id']}  "
                   f"off selama {fmt(miss_s)}", 'cy','b')

        # Expire → siapa ID 1 baru
        for exp in expired_log:
            miss_s   = now - exp.get('last_seen', now)
            on_dur   = exp.get('last_on', exp['enter_time']) - exp['enter_time']
            total_off= exp.get('total_off', 0.0) + miss_s
            sep()
            cp(f"  [{ts()}]  x EXPIRE   ID {exp['display_id']}  "
               f"| on {fmt(on_dur)}  off {fmt(total_off)}", 'rd','b')
            new_id1 = next((t for t in tracks.values() if t['display_id']==1), None)
            if new_id1:
                cp(f"         L ID 1 sekarang -> masuk "
                   f"{time.strftime('%H:%M:%S',time.localtime(new_id1['enter_time']))}  "
                   f"aktif {fmt(now-new_id1['enter_time'])}  "
                   f"lebar {new_id1['width_mm']:.1f}mm", 'b','cy')
            else:
                cp("         L Tidak ada objek aktif.", 'gy')
            remaining = sorted(tracks.values(), key=lambda t: t['display_id'])
            if remaining:
                cp("         Antrian: " + "  ".join(
                   f"ID{t['display_id']}({fmt(now-t['enter_time'])})"
                   for t in remaining), 'gy')
            sep()

        # Cleanup
        alive = set(tracks.keys())
        for d in [k for k in list(self._state)      if k not in alive]: del self._state[d]
        for d in [k for k in list(self._did_map)    if k not in alive]: del self._did_map[d]
        for d in [k for k in list(self._miss_start) if k not in alive]: del self._miss_start[d]
        for d in [k for k in list(self._last_tick)  if k not in alive]: del self._last_tick[d]

# ─────────────────────────────────────────────
#  ANOTASI FRAME
# ─────────────────────────────────────────────
COLORS = [(0,220,255),(0,255,120),(255,180,0),(200,80,255),(255,80,80),(80,180,255)]

def draw_tracks(frame, tracks):
    now = time.time()
    for _, t in tracks.items():
        did    = t['display_id']
        color  = COLORS[(did-1) % len(COLORS)]
        x1,y1,x2,y2 = [int(v) for v in t['bbox']]
        thick  = 3 if did==1 else 2
        miss   = t.get('missing', False)
        miss_s = (now - t['last_seen']) if miss else 0.0

        # Durasi ON frame (waktu aktif terdeteksi)
        total_off = t.get('total_off', 0.0)
        if miss and t.get('off_time'):
            total_off += now - t['off_time']
        on_dur = (now - t['enter_time']) - total_off

        # Durasi OFF frame saat ini
        off_cur = miss_s  # detik sejak terakhir terdeteksi

        enter_str = time.strftime('%H:%M:%S', time.localtime(t['enter_time']))

        # ── Bbox: solid atau putus-putus
        if miss:
            dash, gap = 12, 6
            for p1,p2 in [((x1,y1),(x2,y1)),((x2,y1),(x2,y2)),
                           ((x2,y2),(x1,y2)),((x1,y2),(x1,y1))]:
                tot = int(np.hypot(p2[0]-p1[0],p2[1]-p1[1]))
                for s in range(0, tot, dash+gap):
                    ex=int(p1[0]+(p2[0]-p1[0])*min(s+dash,tot)/tot)
                    ey=int(p1[1]+(p2[1]-p1[1])*min(s+dash,tot)/tot)
                    sx=int(p1[0]+(p2[0]-p1[0])*s/tot)
                    sy=int(p1[1]+(p2[1]-p1[1])*s/tot)
                    cv2.line(frame,(sx,sy),(ex,ey),color,thick)
        else:
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,thick)

        # ── Badge ID pojok kiri atas bbox
        id_txt   = f"ID {did}"
        id_scale = 0.85 if did==1 else 0.70
        (iw,ih),_ = cv2.getTextSize(id_txt, cv2.FONT_HERSHEY_DUPLEX, id_scale, 2)
        pad = 7
        bx2,by2 = x1+iw+pad*2, y1+ih+pad*2
        cv2.rectangle(frame,(x1,y1),(bx2,by2),color,-1)
        if did == 1:
            cv2.rectangle(frame,(x1,y1),(bx2,by2),(255,255,255),1)
        cv2.putText(frame, id_txt, (x1+pad, y1+ih+pad-1),
                    cv2.FONT_HERSHEY_DUPLEX, id_scale, (255,255,255), 2, cv2.LINE_AA)

        # ── Garis ukuran lebar di bawah bbox
        my = y2 + 10
        cv2.line(frame,(x1,my),(x2,my),color,1)
        cv2.line(frame,(x1,my-5),(x1,my+5),color,2)
        cv2.line(frame,(x2,my-5),(x2,my+5),color,2)
        cv2.putText(frame, f"{t['width_mm']:.0f}mm",
                    ((x1+x2)//2-26, my+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)

        # ── Info DI ATAS bbox (di atas garis atas bbox)
        # Susunan dari atas ke bawah:
        #   [a] Waktu masuk frame (enter_time)
        #   [b] Durasi ON frame   (total waktu aktif terdeteksi)
        #   [c] Durasi OFF frame  (saat ini sedang hilang — hanya jika missing)

        info_rows = [
            (f"Masuk : {enter_str}",           color,           (0,0,0)),
            (f"ON    : {fmt(on_dur)}",          (0,180,60),      (255,255,255)),
        ]
        if miss and off_cur < GRACE_SEC:
            remain = max(0.0, GRACE_SEC - off_cur)
            info_rows.append(
                (f"OFF   : {off_cur:.2f}s  sisa {remain:.1f}s",
                 (0,100,210), (255,255,255))
            )

        # Gambar dari bawah ke atas tepat di atas y1
        line_h = 18
        for i, (lbl, bg, tc) in enumerate(reversed(info_rows)):
            (tw,th),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            by = y1 - 6 - i * (line_h + 2)
            if by < th + 6:
                # Tidak cukup ruang di atas — pindah ke bawah bbox
                by = y2 + 36 + i * (line_h + 2)
            cv2.rectangle(frame,(x1, by-th-2),(x1+tw+4, by+2), bg, -1)
            cv2.putText(frame, lbl, (x1+2, by),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, tc, 1, cv2.LINE_AA)

        # Centroid
        cx,cy = centroid(t['bbox'])
        cv2.circle(frame,(cx,cy),4,color,-1)


def draw_rejected_subtle(frame, bbox, w_mm):
    """Tampilkan outline sangat tipis untuk bbox di luar range — tidak mengganggu."""
    x1,y1,x2,y2 = [int(v) for v in bbox]
    cv2.rectangle(frame,(x1,y1),(x2,y2),(50,50,50),1)
    lbl = (f"{w_mm:.0f}<{WIDTH_MIN_MM:.0f}mm" if w_mm < WIDTH_MIN_MM
           else f"{w_mm:.0f}>{WIDTH_MAX_MM:.0f}mm")
    cv2.putText(frame, lbl, (x1+2,y1-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (80,80,80), 1, cv2.LINE_AA)


def draw_hud(frame, tracks, n_rej):
    """Overlay HUD di pojok kiri atas."""
    now  = time.time()
    h, w = frame.shape[:2]

    # Bar atas
    cv2.rectangle(frame,(0,0),(w,34),(15,15,15),-1)
    cv2.putText(frame,
                f"Aktif: {len(tracks)}  |  Ditolak: {n_rej}  |  "
                f"Range: {WIDTH_MIN_MM:.0f}-{WIDTH_MAX_MM:.0f}mm  |  "
                f"Grace: {GRACE_SEC:.0f}s",
                (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200,200,200), 1, cv2.LINE_AA)

    # Bar bawah — daftar semua ID aktif + durasi on
    if tracks:
        parts = []
        for t in sorted(tracks.values(), key=lambda x: x['display_id']):
            total_off = t.get('total_off',0.0)
            if t.get('missing') and t.get('off_time'):
                total_off += now - t['off_time']
            on = (now - t['enter_time']) - total_off
            parts.append(f"ID{t['display_id']} on:{fmt(on)}")
        bar = "  |  ".join(parts)
        cv2.rectangle(frame,(0,h-28),(w,h),(15,15,15),-1)
        cv2.putText(frame, bar, (8,h-9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    sep()
    cp("  RT-DETR  Car Width Tracker  |  Fixed Range  |  No Kalibrasi",'b','wh')
    cp(f"  Range valid : {WIDTH_MIN_MM:.0f} – {WIDTH_MAX_MM:.0f} mm",'cy')
    cp(f"  PIXEL_PER_MM: {PIXEL_PER_MM}  (ubah di bagian KONFIGURASI jika perlu)",'gy')
    cp(f"  Grace period: {GRACE_SEC:.0f} detik  — lebih dari {GRACE_SEC:.0f}s hilang → bbox dihapus",'gy')
    cp("  Tekan 'Q' untuk keluar  |  'R' untuk reset track",'gy')
    sep()

    model     = RTDETR('rtdetr-l.pt')
    manager   = IDManager()
    ev_logger = EventLogger()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera tidak ditemukan."); return

    WIN = "Car Tracker — Fixed Range 1600-1850mm"
    cv2.namedWindow(WIN)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break

        # ── Deteksi RT-DETR
        raw = list(model(frame, stream=True))
        valid_dets = []
        rej_dets   = []

        for r in raw:
            for box in r.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                if cls not in CAR_CLASS_IDS or conf < CONF_THRESHOLD:
                    continue
                x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
                bbox = [x1,y1,x2,y2]
                valid, w_mm = is_valid_width(bbox)

                if valid:
                    valid_dets.append({
                        'bbox':     bbox,
                        'conf':     conf,
                        'width_mm': w_mm,
                    })
                else:
                    rej_dets.append({'bbox': bbox, 'width_mm': w_mm})

        # Gambar bbox ditolak (sangat tipis, tidak mengganggu)
        for rd in rej_dets:
            draw_rejected_subtle(frame, rd['bbox'], rd['width_mm'])

        # Update tracker
        tracks, expired = manager.update(frame, valid_dets)

        # Log terminal
        ev_logger.update(tracks, expired)

        # Gambar hasil
        draw_tracks(frame, tracks)
        draw_hud(frame, tracks, len(rej_dets))

        cv2.imshow(WIN, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')): break
        if key in (ord('r'), ord('R')):
            manager   = IDManager()
            ev_logger = EventLogger()
            sep(); cp("  Track di-reset.", 'yw'); sep()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()