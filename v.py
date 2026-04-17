import cv2
import numpy as np
import time
from collections import OrderedDict
from ultralytics import RTDETR

# ─────────────────────────────────────────────
#  KONFIGURASI
# ─────────────────────────────────────────────
CAR_CLASS_IDS  = {2, 5, 7}
CONF_THRESHOLD = 0.45
WIDTH_MIN_MM   = 1600.0
WIDTH_MAX_MM   = 1855.0

W_IOU      = 0.50
W_DIST     = 0.30
W_APP      = 0.20
MIN_MATCH  = 0.30
GRACE_SEC  = 9.9
MAX_DIST   = 400

# ─────────────────────────────────────────────
#  ANSI TERMINAL
# ─────────────────────────────────────────────
A = {'rst':'\033[0m','b':'\033[1m','cy':'\033[96m','gr':'\033[92m',
     'yw':'\033[93m','rd':'\033[91m','gy':'\033[90m','wh':'\033[97m'}
def cp(txt,*s): print(''.join(A.get(x,'') for x in s)+txt+A['rst'])
def sep(): cp("─"*64,'gy')
def ts(): return time.strftime('%H:%M:%S')
def fmt(sec):
    if sec<60: return f"{sec:06.3f}s"
    m=int(sec//60); return f"{m}:{sec-m*60:06.3f}s"


# ─────────────────────────────────────────────
#  KALIBRASI — input di terminal saat startup
#  Tidak perlu tekan tombol di jendela kamera
# ─────────────────────────────────────────────
class Calibrator:
    def __init__(self):
        self.pixel_per_mm  = None
        self.mode          = 'idle'   # 'idle' | 'wait_click' | 'done'
        self.points        = []
        self._preview_frame = None

    # ── Dipanggil dari terminal sebelum loop utama
    def setup_terminal(self, cap):
        """
        Tiga opsi input yang ditawarkan ke user:
        1. Klik 2 titik di frame preview (mode klik)
        2. Input langsung pixel + mm yang sudah diukur
        3. Input pixel_per_mm langsung jika sudah diketahui
        """
        sep()
        cp("  SETUP KALIBRASI", 'b','wh')
        sep()
        cp("  Pilih metode kalibrasi:", 'cy')
        cp("  [1] Klik 2 titik di jendela preview (buka frame dulu)", 'wh')
        cp("  [2] Input manual: ukur pixel bbox & mm nyata", 'wh')
        cp("  [3] Input langsung nilai px/mm (jika sudah diketahui)", 'wh')
        cp("  [4] Lewati kalibrasi — gunakan estimasi default (kurang akurat)", 'gy')
        sep()

        while True:
            try:
                pilihan = input("  Pilih [1/2/3/4]: ").strip()
            except (EOFError, KeyboardInterrupt):
                pilihan = '4'

            if pilihan == '1':
                self._mode_klik(cap)
                break
            elif pilihan == '2':
                self._mode_manual_px_mm()
                break
            elif pilihan == '3':
                self._mode_langsung()
                break
            elif pilihan == '4':
                self._mode_default()
                break
            else:
                cp("  Input tidak valid, ketik 1/2/3/4.", 'yw')

    # ── METODE 1: klik di jendela preview ──
    def _mode_klik(self, cap):
        cp("\n  Jendela preview akan terbuka.", 'cy')
        cp("  Klik TEPAT di ujung KIRI objek referensi, lalu ujung KANAN.", 'cy')
        cp("  Setelah 2 klik selesai, masukkan jarak nyata di terminal.", 'cy')
        cp("  Tekan 'Q' di jendela preview untuk batal.\n", 'gy')

        ok, frame = cap.read()
        if not ok:
            cp("  Gagal membaca frame kamera.", 'rd')
            self._mode_default(); return

        self.points = []
        self.mode   = 'wait_click'
        WIN_CAL     = "KALIBRASI — klik 2 titik horizontal"
        cv2.namedWindow(WIN_CAL)
        cv2.setMouseCallback(WIN_CAL, self._mouse_cb)

        # Tampilkan frame diam (bukan live) agar mudah diklik tepat
        display = frame.copy()
        h, w    = display.shape[:2]
        cv2.putText(display,
                    "Klik ujung KIRI lalu ujung KANAN objek referensi",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(display, "Tekan Q untuk batal",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (160, 160, 160), 1, cv2.LINE_AA)

        while self.mode == 'wait_click':
            show = display.copy()

            # Gambar titik yang sudah diklik
            for i, (px, py) in enumerate(self.points):
                cv2.circle(show, (px, py), 7, (0, 220, 255), -1)
                cv2.circle(show, (px, py), 7, (255,255,255), 1)
                cv2.putText(show, f"P{i+1}", (px+10, py-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 220, 255), 2, cv2.LINE_AA)
            if len(self.points) == 2:
                cv2.line(show, self.points[0], self.points[1],
                         (0, 220, 255), 2)
                px_dist = abs(self.points[1][0] - self.points[0][0])
                cv2.putText(show, f"{px_dist} px",
                            ((self.points[0][0]+self.points[1][0])//2 - 20,
                             self.points[0][1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(WIN_CAL, show)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), ord('Q')):
                cp("  Kalibrasi dibatalkan.", 'yw')
                cv2.destroyWindow(WIN_CAL)
                self._mode_default(); return

        cv2.destroyWindow(WIN_CAL)

        # Setelah 2 titik diklik, minta input mm di terminal
        if len(self.points) == 2:
            px_dist = abs(self.points[1][0] - self.points[0][0])
            sep()
            cp(f"  Jarak pixel terukur: {px_dist} px", 'cy')
            while True:
                try:
                    real_mm = float(input("  Masukkan lebar nyata objek referensi (mm): "))
                    if real_mm <= 0: raise ValueError
                    break
                except ValueError:
                    cp("  Input tidak valid, masukkan angka positif.", 'yw')
            self.pixel_per_mm = px_dist / real_mm
            self._print_result()

    def _mouse_cb(self, event, x, y, flags, param):
        if self.mode != 'wait_click': return
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 2:
                self.points.append((x, y))
                cp(f"  Titik {len(self.points)}: ({x}, {y})", 'cy')
            if len(self.points) == 2:
                self.mode = 'done'

    # ── METODE 2: input pixel bbox + mm ──
    def _mode_manual_px_mm(self):
        sep()
        cp("  Cara ukur pixel bbox:", 'cy')
        cp("  Jalankan program sebentar, lihat angka lebar px di bbox mobil referensi.", 'gy')
        cp("  Atau pause video dan ukur di image editor.", 'gy')
        sep()
        while True:
            try:
                px = float(input("  Lebar bbox mobil referensi di layar (pixel): "))
                mm = float(input("  Lebar nyata mobil referensi (mm): "))
                if px <= 0 or mm <= 0: raise ValueError
                break
            except ValueError:
                cp("  Input tidak valid, coba lagi.", 'yw')
        self.pixel_per_mm = px / mm
        self._print_result()

    # ── METODE 3: input px/mm langsung ──
    def _mode_langsung(self):
        sep()
        while True:
            try:
                ppm = float(input("  Masukkan nilai pixel_per_mm: "))
                if ppm <= 0: raise ValueError
                break
            except ValueError:
                cp("  Input tidak valid.", 'yw')
        self.pixel_per_mm = ppm
        self._print_result()

    # ── METODE 4: default / skip ──
    def _mode_default(self):
        # Estimasi kasar: asumsi lebar frame ~5 meter di jarak 10 meter
        # Ini hanya fallback, tidak akurat
        self.pixel_per_mm = None
        sep()
        cp("  Kalibrasi dilewati.", 'yw')
        cp("  Validasi lebar mm tidak aktif — semua mobil yang terdeteksi akan ditampilkan.", 'yw')
        sep()

    def _print_result(self):
        sep()
        cp(f"  Kalibrasi OK: {self.pixel_per_mm:.4f} px/mm", 'gr','b')
        lo = int(WIDTH_MIN_MM * self.pixel_per_mm)
        hi = int(WIDTH_MAX_MM * self.pixel_per_mm)
        cp(f"  Bbox width valid: {lo} – {hi} px  ({WIDTH_MIN_MM:.0f}–{WIDTH_MAX_MM:.0f} mm)", 'cy')
        sep()

    def px_to_mm(self, px):
        return px / self.pixel_per_mm if self.pixel_per_mm else None

    def is_calibrated(self):
        return self.pixel_per_mm is not None

    # ── Overlay di frame saat tracking (hanya info, tidak ada interaksi) ──
    def draw_info(self, frame):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h-28), (w, h), (15,15,15), -1)
        if not self.is_calibrated():
            msg = "Kalibrasi tidak aktif — semua mobil ditampilkan"
            col = (0, 140, 255)
        else:
            lo  = int(WIDTH_MIN_MM * self.pixel_per_mm)
            hi  = int(WIDTH_MAX_MM * self.pixel_per_mm)
            msg = (f"Kalibrasi: {self.pixel_per_mm:.4f} px/mm  |  "
                   f"valid: {lo}–{hi} px  ({WIDTH_MIN_MM:.0f}–{WIDTH_MAX_MM:.0f} mm)")
            col = (0, 200, 100)
        cv2.putText(frame, msg, (8, h-9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
#  VALIDASI LEBAR
# ─────────────────────────────────────────────
def validate_width(bbox, calib):
    if not calib.is_calibrated():
        # Tanpa kalibrasi: loloskan semua (tidak ada filter lebar)
        w_px = bbox[2] - bbox[0]
        return True, w_px, 'no_calib'
    w_mm = calib.px_to_mm(bbox[2] - bbox[0])
    if w_mm < WIDTH_MIN_MM: return False, w_mm, 'terlalu_kecil'
    if w_mm > WIDTH_MAX_MM: return False, w_mm, 'terlalu_besar'
    return True, w_mm, 'ok'


# ─────────────────────────────────────────────
#  UTILITAS TRACKING
# ─────────────────────────────────────────────
def iou(b1, b2):
    ix1=max(b1[0],b2[0]); iy1=max(b1[1],b2[1])
    ix2=min(b1[2],b2[2]); iy2=min(b1[3],b2[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    u=(b1[2]-b1[0])*(b1[3]-b1[1])+(b2[2]-b2[0])*(b2[3]-b2[1])-inter
    return inter/u if u>0 else 0.0

def centroid(b): return ((b[0]+b[2])//2, (b[1]+b[3])//2)

def ndist(b1, b2):
    c1=centroid(b1); c2=centroid(b2)
    return min(np.hypot(c1[0]-c2[0],c1[1]-c2[1])/MAX_DIST, 1.0)

def get_appearance(frame, bbox):
    x1,y1,x2,y2 = [max(0,int(v)) for v in bbox]
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return np.zeros(48)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv],[0],None,[16],[0,180]).flatten()
    s = cv2.calcHist([hsv],[1],None,[16],[0,256]).flatten()
    v = cv2.calcHist([hsv],[2],None,[16],[0,256]).flatten()
    hist = np.concatenate([h,s,v]).astype(np.float32)
    n = hist.sum(); return hist/n if n>0 else hist

def app_sim(h1, h2):
    d=np.dot(h1,h2); n=np.linalg.norm(h1)*np.linalg.norm(h2)
    return float(d/n) if n>0 else 0.0

def mscore(track, bbox, app):
    return (W_IOU*iou(track['bbox'],bbox)
           +W_DIST*(1-ndist(track['bbox'],bbox))
           +W_APP*app_sim(track['appearance'],app))


# ─────────────────────────────────────────────
#  TERMINAL EVENT LOGGER
# ─────────────────────────────────────────────
class EventLogger:
    def __init__(self):
        self._state={};self._did_map={};self._miss_start={};self._last_tick={}

    def update(self, tracks, expired_log):
        now=time.time()
        for iid,t in tracks.items():
            if iid not in self._did_map:
                sep()
                cp(f"  [{ts()}]  + MASUK    ID {t['display_id']}  "
                   f"lebar {t['width_mm']:.1f}mm  conf {t['conf']:.2f}",'gr','b')
                self._did_map[iid]=t['display_id']; self._state[iid]='active'
        for iid,t in tracks.items():
            prev=self._did_map.get(iid)
            if prev is not None and t['display_id']!=prev:
                sep()
                cp(f"  [{ts()}]  ^ PROMOSI  ID {prev} -> ID {t['display_id']}  "
                   f"aktif {fmt(now-t['enter_time'])}",'cy','b')
                self._did_map[iid]=t['display_id']
        for iid,t in tracks.items():
            if t['missing'] and self._state.get(iid)=='active':
                self._state[iid]='missing'
                self._miss_start[iid]=now; self._last_tick[iid]=now
                sep()
                cp(f"  [{ts()}]  o HILANG   ID {t['display_id']}  "
                   f"grace {GRACE_SEC:.1f}s",'yw','b')
            elif t['missing'] and self._state.get(iid)=='missing':
                miss_dur=now-self._miss_start.get(iid,now)
                if now-self._last_tick.get(iid,now)>=1.0:
                    remain=max(0.0,GRACE_SEC-miss_dur)
                    cp(f"         | ID {t['display_id']}  hilang {miss_dur:.1f}s  "
                       f"sisa {remain:.1f}s",'yw')
                    self._last_tick[iid]=now
            elif not t['missing'] and self._state.get(iid)=='missing':
                miss_dur=now-self._miss_start.get(iid,now)
                self._state[iid]='active'
                sep()
                cp(f"  [{ts()}]  < KEMBALI  ID {t['display_id']}  "
                   f"hilang {fmt(miss_dur)}",'cy','b')
        for exp in expired_log:
            miss_dur=now-exp.get('last_seen',now)
            sep()
            cp(f"  [{ts()}]  x EXPIRE   ID {exp['display_id']}  "
               f"total {fmt(now-exp['enter_time'])}  hilang {fmt(miss_dur)}",'rd','b')
            new_id1=next((t for t in tracks.values() if t['display_id']==1),None)
            if new_id1:
                cp(f"         L ID 1 sekarang -> masuk "
                   f"{time.strftime('%H:%M:%S',time.localtime(new_id1['enter_time']))}  "
                   f"aktif {fmt(now-new_id1['enter_time'])}  "
                   f"lebar {new_id1['width_mm']:.1f}mm",'b','cy')
            else:
                cp("         L Tidak ada objek aktif.",'gy')
            remaining=sorted(tracks.values(),key=lambda t:t['display_id'])
            if remaining:
                cp("         Antrian: "+"  ".join(
                    f"ID{t['display_id']}({fmt(now-t['enter_time'])})"
                    for t in remaining),'gy')
            sep()
        alive=set(tracks.keys())
        for d in [k for k in list(self._state) if k not in alive]:
            del self._state[d]
        for d in [k for k in list(self._did_map) if k not in alive]:
            del self._did_map[d]
        for d in [k for k in list(self._miss_start) if k not in alive]:
            del self._miss_start[d]
        for d in [k for k in list(self._last_tick) if k not in alive]:
            del self._last_tick[d]


# ─────────────────────────────────────────────
#  ID MANAGER
# ─────────────────────────────────────────────
class IDManager:
    def __init__(self):
        self.tracks=OrderedDict(); self._ctr=0

    def _rank(self):
        for rank,(_,t) in enumerate(self.tracks.items(),start=1):
            t['display_id']=rank

    def update(self, frame, detections):
        now=time.time()
        for d in detections:
            d['appearance']=get_appearance(frame,d['bbox'])
        used,matched={},{}
        for tid,track in self.tracks.items():
            bs,bi=-1,-1
            for i,d in enumerate(detections):
                if i in used: continue
                s=mscore(track,d['bbox'],d['appearance'])
                if s>bs: bs,bi=s,i
            if bs>=MIN_MATCH:
                matched[tid]=bi; used[bi]=True
        for tid,di in matched.items():
            d=detections[di]; t=self.tracks[tid]
            t['bbox']=d['bbox']; t['conf']=d['conf']
            t['width_mm']=d['width_mm']
            t['appearance']=d['appearance']
            t['last_seen']=now; t['missing']=False
        for tid in self.tracks:
            if tid not in matched: self.tracks[tid]['missing']=True
        for i,d in enumerate(detections):
            if i not in used:
                self.tracks[self._ctr]={
                    'bbox':d['bbox'],'conf':d['conf'],
                    'width_mm':d['width_mm'],
                    'appearance':d['appearance'],
                    'enter_time':now,'last_seen':now,
                    'missing':False,'display_id':None,
                }
                self._ctr+=1
        self._rank()
        expired=[dict(t) for _,t in self.tracks.items()
                 if t['missing'] and now-t['last_seen']>GRACE_SEC]
        for tid in [k for k,t in self.tracks.items()
                    if t['missing'] and now-t['last_seen']>GRACE_SEC]:
            del self.tracks[tid]
        self._rank()
        return self.tracks, expired


# ─────────────────────────────────────────────
#  ANOTASI FRAME
# ─────────────────────────────────────────────
COLORS=[(0,220,255),(0,255,120),(255,180,0),(200,80,255),(255,80,80),(80,180,255)]

def draw_tracks(frame, tracks, calib):
    now=time.time()
    for _,t in tracks.items():
        did   = t['display_id']
        color = COLORS[(did-1)%len(COLORS)]
        x1,y1,x2,y2=[int(v) for v in t['bbox']]
        thick = 3 if did==1 else 2
        miss  = t.get('missing',False)
        miss_s= (now-t['last_seen']) if miss else 0.0
        dur   = now-t['enter_time']
        enter = time.strftime('%H:%M:%S',time.localtime(t['enter_time']))

        # Bbox solid atau putus-putus
        if miss:
            dash,gap=12,6
            for p1,p2 in [((x1,y1),(x2,y1)),((x2,y1),(x2,y2)),
                           ((x2,y2),(x1,y2)),((x1,y2),(x1,y1))]:
                tot=int(np.hypot(p2[0]-p1[0],p2[1]-p1[1]))
                for s in range(0,tot,dash+gap):
                    ex=int(p1[0]+(p2[0]-p1[0])*min(s+dash,tot)/tot)
                    ey=int(p1[1]+(p2[1]-p1[1])*min(s+dash,tot)/tot)
                    sx=int(p1[0]+(p2[0]-p1[0])*s/tot)
                    sy=int(p1[1]+(p2[1]-p1[1])*s/tot)
                    cv2.line(frame,(sx,sy),(ex,ey),color,thick)
        else:
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,thick)

        # Badge ID pojok kiri atas
        id_txt   = f"ID {did}"
        id_scale = 0.85 if did==1 else 0.70
        (iw,ih),_ = cv2.getTextSize(id_txt,cv2.FONT_HERSHEY_DUPLEX,id_scale,2)
        pad=7; bx2,by2=x1+iw+pad*2,y1+ih+pad*2
        cv2.rectangle(frame,(x1,y1),(bx2,by2),color,-1)
        if did==1:
            cv2.rectangle(frame,(x1,y1),(bx2,by2),(255,255,255),1)
        cv2.putText(frame,id_txt,(x1+pad,y1+ih+pad-1),
                    cv2.FONT_HERSHEY_DUPLEX,id_scale,(255,255,255),2,cv2.LINE_AA)

        # Garis ukuran lebar + nilai mm/px
        my=y2+10
        cv2.line(frame,(x1,my),(x2,my),color,1)
        cv2.line(frame,(x1,my-5),(x1,my+5),color,2)
        cv2.line(frame,(x2,my-5),(x2,my+5),color,2)
        if calib.is_calibrated():
            w_label=f"{t['width_mm']:.0f}mm"
        else:
            w_label=f"{x2-x1}px"
        cv2.putText(frame,w_label,((x1+x2)//2-22,my+18),
                    cv2.FONT_HERSHEY_SIMPLEX,0.44,color,1,cv2.LINE_AA)

        # Label info di bawah bbox
        rows=[
            f"conf:{t['conf']:.2f}  {t['width_mm']:.1f}{'mm' if calib.is_calibrated() else 'px'}",
            f"In: {enter}",
            f"Dur: {fmt(dur)}",
        ]
        if miss and miss_s<10.0:
            remain=max(0.0,GRACE_SEC-miss_s)
            rows.append(f"Hilang: {miss_s:.2f}s  sisa {remain:.1f}s")

        for row,lbl in enumerate(rows):
            (tw,th),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.42,1)
            by=y2+22+row*(th+5)
            is_miss=(miss and row==len(rows)-1)
            bg=(0,90,200) if is_miss else color
            tc=(255,255,255) if is_miss else (0,0,0)
            cv2.rectangle(frame,(x1,by-th-2),(x1+tw+4,by+2),bg,-1)
            cv2.putText(frame,lbl,(x1+2,by),
                        cv2.FONT_HERSHEY_SIMPLEX,0.42,tc,1,cv2.LINE_AA)

        cx,cy=centroid(t['bbox'])
        cv2.circle(frame,(cx,cy),4,color,-1)


def draw_rejected(frame, bbox, w_mm, status, calib):
    x1,y1,x2,y2=[int(v) for v in bbox]
    cv2.rectangle(frame,(x1,y1),(x2,y2),(60,60,180),1)
    if w_mm and calib.is_calibrated():
        lbl=(f"{w_mm:.0f}mm<{WIDTH_MIN_MM:.0f}" if status=='terlalu_kecil'
             else f"{w_mm:.0f}mm>{WIDTH_MAX_MM:.0f}")
        cv2.putText(frame,lbl,(x1+2,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,(80,80,200),1,cv2.LINE_AA)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    # Buka kamera DULU sebelum setup kalibrasi
    # (agar bisa ambil frame untuk mode klik)
    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera tidak ditemukan."); return

    sep()
    cp("  RT-DETR  Car Width Tracker  |  ID Konsisten",'b','wh')
    cp(f"  Range: {WIDTH_MIN_MM:.0f}–{WIDTH_MAX_MM:.0f} mm  "
       f"Grace: {GRACE_SEC}s",'gy')
    sep()

    # ── Kalibrasi di terminal SEBELUM loop kamera ──
    calib=Calibrator()
    calib.setup_terminal(cap)

    model     = RTDETR('rtdetr-l.pt')
    manager   = IDManager()
    ev_logger = EventLogger()

    WIN="Car Width Tracker — RT-DETR"
    cv2.namedWindow(WIN)

    sep()
    cp("  Tracking dimulai. Tekan 'Q' di jendela kamera untuk keluar.",'gr','b')
    cp("  Tekan 'R' di jendela kamera untuk reset semua track.",'gy')
    sep()

    while cap.isOpened():
        ok,frame=cap.read()
        if not ok: break

        raw=list(model(frame,stream=True))
        valid_dets,rej_dets=[],[]

        for r in raw:
            for box in r.boxes:
                cls=int(box.cls[0]); conf=float(box.conf[0])
                if cls not in CAR_CLASS_IDS or conf<CONF_THRESHOLD: continue
                x1,y1,x2,y2=[int(v) for v in box.xyxy[0].tolist()]
                bbox=[x1,y1,x2,y2]
                ok_w,w_mm,status=validate_width(bbox,calib)
                if ok_w:
                    valid_dets.append({'bbox':bbox,'conf':conf,'width_mm':w_mm or 0})
                else:
                    rej_dets.append({'bbox':bbox,'width_mm':w_mm,'status':status})

        for rd in rej_dets:
            draw_rejected(frame,rd['bbox'],rd['width_mm'],rd['status'],calib)

        tracks,expired=manager.update(frame,valid_dets)
        ev_logger.update(tracks,expired)
        draw_tracks(frame,tracks,calib)

        cv2.putText(frame,
                    f"Aktif: {len(tracks)}  Ditolak: {len(rej_dets)}",
                    (10,28),cv2.FONT_HERSHEY_SIMPLEX,0.6,
                    (255,255,255),2,cv2.LINE_AA)

        calib.draw_info(frame)
        cv2.imshow(WIN,frame)

        # Tombol hanya perlu fokus di jendela OpenCV
        key=cv2.waitKey(1)&0xFF
        if key in(ord('q'),ord('Q')): break
        if key in(ord('r'),ord('R')):
            manager=IDManager(); ev_logger=EventLogger()
            sep(); cp("  Track di-reset.",'yw'); sep()

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()