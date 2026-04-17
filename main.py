import cv2
import numpy as np
import time
import urllib.request
import os
from collections import OrderedDict
from ultralytics import RTDETR

# ─────────────────────────────────────────────
#  KONFIGURASI
# ─────────────────────────────────────────────
PERSON_CLASS_ID  = 0
CONF_THRESHOLD   = 0.45
FACE_RATIO_TOP   = 0.28
FACE_CONFIRM_MIN = 0.0

W_IOU        = 0.50
W_DIST       = 0.30
W_APP        = 0.20
MIN_MATCH    = 0.30
GRACE_SEC    = 9.9    # grace period maks 9.9 detik (< 10 detik)
MAX_DIST     = 300

PROTO_PATH = "deploy.prototxt"
MODEL_PATH = "face_detector.caffemodel"
PROTO_URL  = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL  = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"


# ─────────────────────────────────────────────
#  LOG TERMINAL — warna ANSI
# ─────────────────────────────────────────────
ANSI = {
    'reset':  '\033[0m',
    'bold':   '\033[1m',
    'cyan':   '\033[96m',
    'green':  '\033[92m',
    'yellow': '\033[93m',
    'red':    '\033[91m',
    'magenta':'\033[95m',
    'gray':   '\033[90m',
    'white':  '\033[97m',
}

def cprint(text, *styles):
    prefix = ''.join(ANSI.get(s, '') for s in styles)
    print(f"{prefix}{text}{ANSI['reset']}")

def log_separator():
    cprint("─" * 58, 'gray')

def log_event(msg, style='white'):
    ts = time.strftime('%H:%M:%S')
    cprint(f"  [{ts}]  {msg}", style)


# ─────────────────────────────────────────────
#  EVENT TRACKER — catat pergantian ID di terminal
# ─────────────────────────────────────────────
class EventLogger:
    """
    Melacak status setiap display-ID dan mencetak ke terminal:
    - Objek baru masuk
    - Objek hilang (mulai grace period)
    - Objek kembali (dalam grace period)
    - Objek expire → ID berikutnya naik posisi
    """
    def __init__(self):
        # display_id -> state: 'active' | 'missing' | 'expired'
        self._state   = {}
        # internal_id -> display_id terakhir yang diketahui
        self._did_map = {}
        # posisi ID 1 sebelumnya (internal_id)
        self._prev_id1_internal = None

    def update(self, tracks, expired_log):
        """
        tracks      : dict {internal_id: track_info}  — hasil IDManager.update()
        expired_log : list of track_info yang baru saja di-expire
        """
        now = time.time()

        # — Cek objek baru masuk
        for iid, t in tracks.items():
            did = t['display_id']
            if iid not in self._did_map:
                log_separator()
                log_event(
                    f"🟢  Objek baru  |  ID {did}  |  masuk {time.strftime('%H:%M:%S', time.localtime(t['enter_time']))}",
                    'green'
                )
                self._did_map[iid] = did

        # — Cek perubahan display_id (naik posisi karena ID sebelumnya expire)
        for iid, t in tracks.items():
            did = t['display_id']
            prev_did = self._did_map.get(iid)
            if prev_did is not None and did != prev_did:
                log_separator()
                log_event(
                    f"⬆️   Promosi ID  |  ID {prev_did} → ID {did}  "
                    f"(objek telah ada {format_dur(now - t['enter_time'])})",
                    'cyan'
                )
                self._did_map[iid] = did

        # — Cek objek yang mulai hilang (grace period dimulai)
        for iid, t in tracks.items():
            did   = t['display_id']
            state = self._state.get(iid, 'active')
            if t['missing'] and state == 'active':
                self._state[iid] = 'missing'
                log_separator()
                log_event(
                    f"🟡  Hilang       |  ID {did}  |  grace period dimulai  "
                    f"(maks {GRACE_SEC:.1f}s)",
                    'yellow'
                )
            elif not t['missing'] and state == 'missing':
                self._state[iid] = 'active'
                log_separator()
                log_event(
                    f"🔵  Kembali      |  ID {did}  |  terdeteksi lagi",
                    'cyan'
                )

        # — Cek objek yang expire (grace period habis)
        for exp in expired_log:
            exp_did  = exp['display_id']
            exp_dur  = now - exp['enter_time']
            # Siapa yang sekarang menjadi ID 1?
            id1_track = next(
                ((iid, t) for iid, t in tracks.items() if t['display_id'] == 1),
                None
            )
            log_separator()
            cprint(
                f"  🔴  Expire      |  ID {exp_did}  meninggalkan frame  "
                f"|  durasi total {format_dur(exp_dur)}",
                'bold', 'red'
            )
            if id1_track:
                iid1, t1 = id1_track
                cprint(
                    f"       └─ ID 1 sekarang  →  objek yang masuk sejak "
                    f"{time.strftime('%H:%M:%S', time.localtime(t1['enter_time']))}  "
                    f"(sudah {format_dur(now - t1['enter_time'])})",
                    'bold', 'cyan'
                )
            else:
                cprint("       └─ Tidak ada objek aktif saat ini.", 'gray')

        # — Hapus state untuk internal_id yang sudah tidak ada
        active_iids = set(tracks.keys())
        gone = [iid for iid in self._state if iid not in active_iids]
        for iid in gone:
            del self._state[iid]
        gone2 = [iid for iid in self._did_map if iid not in active_iids]
        for iid in gone2:
            del self._did_map[iid]


# ─────────────────────────────────────────────
#  DOWNLOAD DNN FACE MODEL
# ─────────────────────────────────────────────
def download_face_model():
    if not os.path.exists(PROTO_PATH):
        print("Mengunduh prototxt..."); urllib.request.urlretrieve(PROTO_URL, PROTO_PATH)
    if not os.path.exists(MODEL_PATH):
        print("Mengunduh caffemodel..."); urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


# ─────────────────────────────────────────────
#  FACE VALIDATOR
# ─────────────────────────────────────────────
class FaceValidator:
    def __init__(self):
        download_face_model()
        self.net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
        cprint("Face DNN validator siap.", 'green')

    def has_face(self, frame, bbox, min_confidence=0.5):
        x1, y1, x2, y2 = bbox
        h_person = y2 - y1
        face_y2  = min(y1 + int(h_person * FACE_RATIO_TOP), frame.shape[0])
        face_x1, face_y1 = max(x1, 0), max(y1, 0)
        face_x2  = min(x2, frame.shape[1])

        roi = frame[face_y1:face_y2, face_x1:face_x2]
        if roi.shape[0] < 20 or roi.shape[1] < 20:
            return True, [x1, y1, x2, face_y2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(roi, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        dets = self.net.forward()

        best_conf, best_bbox = 0.0, None
        rh, rw = roi.shape[:2]
        for i in range(dets.shape[2]):
            c = float(dets[0, 0, i, 2])
            if c > best_conf:
                best_conf = c
                bx1 = int(dets[0,0,i,3]*rw) + face_x1
                by1 = int(dets[0,0,i,4]*rh) + face_y1
                bx2 = int(dets[0,0,i,5]*rw) + face_x1
                by2 = int(dets[0,0,i,6]*rh) + face_y1
                best_bbox = [bx1, by1, bx2, by2]

        if best_conf >= min_confidence and best_bbox:
            return True, best_bbox
        if FACE_CONFIRM_MIN == 0.0:
            return True, [x1, y1, x2, face_y2]
        return False, None


# ─────────────────────────────────────────────
#  UTILITAS
# ─────────────────────────────────────────────
def compute_iou(b1, b2):
    ix1=max(b1[0],b2[0]); iy1=max(b1[1],b2[1])
    ix2=min(b1[2],b2[2]); iy2=min(b1[3],b2[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]); a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    union=a1+a2-inter
    return inter/union if union>0 else 0.0

def centroid(bbox):
    return ((bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2)

def normalized_dist(b1, b2):
    cx1,cy1=centroid(b1); cx2,cy2=centroid(b2)
    return min(np.hypot(cx1-cx2,cy1-cy2)/MAX_DIST, 1.0)

def extract_appearance(frame, bbox):
    x1,y1,x2,y2=[max(0,int(v)) for v in bbox]
    roi=frame[y1:y2,x1:x2]
    if roi.size==0: return np.zeros(48)
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    h=cv2.calcHist([hsv],[0],None,[16],[0,180]).flatten()
    s=cv2.calcHist([hsv],[1],None,[16],[0,256]).flatten()
    v=cv2.calcHist([hsv],[2],None,[16],[0,256]).flatten()
    hist=np.concatenate([h,s,v]).astype(np.float32)
    n=hist.sum()
    return hist/n if n>0 else hist

def appearance_similarity(h1, h2):
    d=np.dot(h1,h2); n=np.linalg.norm(h1)*np.linalg.norm(h2)
    return float(d/n) if n>0 else 0.0

def match_score(track, det_bbox, det_app):
    iou  = compute_iou(track['bbox'], det_bbox)
    dist = 1.0 - normalized_dist(track['bbox'], det_bbox)
    app  = appearance_similarity(track['appearance'], det_app)
    return W_IOU*iou + W_DIST*dist + W_APP*app

def format_dur(sec):
    if sec < 60: return f"{sec:06.3f}s"
    m=int(sec//60); return f"{m}:{sec-m*60:06.3f}s"


# ─────────────────────────────────────────────
#  ID MANAGER  (mengembalikan expired_log)
# ─────────────────────────────────────────────
class IDManager:
    def __init__(self):
        self.tracks      = OrderedDict()
        self._id_counter = 0

    def _assign_display_ids(self):
        for rank, (_, t) in enumerate(self.tracks.items(), start=1):
            t['display_id'] = rank

    def update(self, frame, detections):
        now = time.time()
        for det in detections:
            det['appearance'] = extract_appearance(frame, det['face_bbox'])

        used_dets, matched = set(), {}
        for tid, track in self.tracks.items():
            best_score, best_idx = -1, -1
            for i, det in enumerate(detections):
                if i in used_dets: continue
                s = match_score(track, det['face_bbox'], det['appearance'])
                if s > best_score: best_score, best_idx = s, i
            if best_score >= MIN_MATCH:
                matched[tid] = best_idx; used_dets.add(best_idx)

        for tid, det_idx in matched.items():
            det = detections[det_idx]; t = self.tracks[tid]
            t['bbox']        = det['face_bbox']
            t['person_bbox'] = det['person_bbox']
            t['conf']        = det['conf']
            t['appearance']  = det['appearance']
            t['last_seen']   = now
            t['missing']     = False

        for tid in self.tracks:
            if tid not in matched:
                self.tracks[tid]['missing'] = True

        for i, det in enumerate(detections):
            if i not in used_dets:
                self.tracks[self._id_counter] = {
                    'bbox':        det['face_bbox'],
                    'person_bbox': det['person_bbox'],
                    'conf':        det['conf'],
                    'appearance':  det['appearance'],
                    'enter_time':  now,
                    'last_seen':   now,
                    'missing':     False,
                    'display_id':  None,
                }
                self._id_counter += 1

        # — Kumpulkan expired sebelum dihapus (untuk log terminal)
        self._assign_display_ids()   # pastikan display_id sudah benar sebelum log
        expired_log = [
            dict(t) for tid, t in self.tracks.items()
            if t['missing'] and (now - t['last_seen']) > GRACE_SEC
        ]
        for tid in [tid for tid, t in self.tracks.items()
                    if t['missing'] and (now - t['last_seen']) > GRACE_SEC]:
            del self.tracks[tid]

        self._assign_display_ids()
        return self.tracks, expired_log


# ─────────────────────────────────────────────
#  ANOTASI FRAME
# ─────────────────────────────────────────────
COLORS = [
    (0,220,255),(0,255,120),(255,180,0),
    (200,80,255),(255,80,80),(80,180,255),
]

def draw_tracks(frame, tracks):
    now = time.time()
    for _, t in tracks.items():
        did   = t['display_id']
        color = COLORS[(did-1) % len(COLORS)]
        dur   = now - t['enter_time']
        enter = time.strftime('%H:%M:%S', time.localtime(t['enter_time']))
        missing = t.get('missing', False)
        missing_sec = (now - t['last_seen']) if missing else 0.0

        fx1,fy1,fx2,fy2 = [int(v) for v in t['bbox']]
        thickness = 3 if did==1 else 2

        # ── Bbox wajah: putus-putus saat missing
        if missing:
            # gambar dashed rectangle manual
            dash_len, gap_len = 10, 6
            pts = [
                ((fx1,fy1),(fx2,fy1)),((fx2,fy1),(fx2,fy2)),
                ((fx2,fy2),(fx1,fy2)),((fx1,fy2),(fx1,fy1)),
            ]
            for (p1,p2) in pts:
                x0,y0=p1; x1e,y1e=p2
                total=int(np.hypot(x1e-x0,y1e-y0))
                for s in range(0,total,dash_len+gap_len):
                    ex=int(x0+(x1e-x0)*min(s+dash_len,total)/total)
                    ey=int(y0+(y1e-y0)*min(s+dash_len,total)/total)
                    sx=int(x0+(x1e-x0)*s/total)
                    sy=int(y0+(y1e-y0)*s/total)
                    cv2.line(frame,(sx,sy),(ex,ey),color,thickness)
        else:
            cv2.rectangle(frame,(fx1,fy1),(fx2,fy2),color,thickness)

        # ── Outline tubuh (semi-transparan)
        px1,py1,px2,py2=[int(v) for v in t['person_bbox']]
        overlay=frame.copy()
        cv2.rectangle(overlay,(px1,py1),(px2,py2),color,1)
        cv2.addWeighted(overlay,0.3,frame,0.7,0,frame)

        # ── Label baris
        labels = [f"ID {did}  conf:{t['conf']:.2f}", f"In: {enter}", f"Dur: {format_dur(dur)}"]

        # Tambah baris durasi hilang jika sedang missing (< 10 detik)
        if missing and missing_sec < 10.0:
            remain = GRACE_SEC - missing_sec
            labels.append(f"Hilang: {missing_sec:.2f}s  (sisa {remain:.1f}s)")

        for row, lbl in enumerate(labels):
            (tw,th),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            by = fy1 - 10 - (len(labels)-1-row)*(th+5)
            if by < th+6: by = fy2+(row+1)*(th+6)
            bg_color = (0,100,200) if (missing and row==len(labels)-1) else color
            cv2.rectangle(frame,(fx1,by-th-2),(fx1+tw+4,by+2),bg_color,-1)
            txt_color = (255,255,255) if (missing and row==len(labels)-1) else (0,0,0)
            cv2.putText(frame,lbl,(fx1+2,by),
                        cv2.FONT_HERSHEY_SIMPLEX,0.42,txt_color,1,cv2.LINE_AA)

        # ── Centroid
        cx,cy=centroid(t['bbox'])
        cv2.circle(frame,(cx,cy),4,color,-1)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    model     = RTDETR('rtdetr-l.pt')
    validator = FaceValidator()
    manager   = IDManager()
    ev_logger = EventLogger()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera tidak ditemukan."); return

    log_separator()
    cprint("  RT-DETR Face Tracker  |  ID Konsisten  |  Grace < 10s", 'bold','white')
    cprint(f"  Grace={GRACE_SEC}s  MinMatch={MIN_MATCH}  Conf>={CONF_THRESHOLD}", 'gray')
    cprint("  Tekan 'q' keluar  |  'r' reset track", 'gray')
    log_separator()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break

        raw = list(model(frame, stream=True))

        person_dets = []
        for r in raw:
            for box in r.boxes:
                cls=int(box.cls[0]); conf=float(box.conf[0])
                if cls != PERSON_CLASS_ID or conf < CONF_THRESHOLD: continue
                x1,y1,x2,y2=[int(v) for v in box.xyxy[0].tolist()]
                person_dets.append({'person_bbox':[x1,y1,x2,y2],'conf':conf})

        face_dets = []
        for pd in person_dets:
            ok_face, face_bbox = validator.has_face(
                frame, pd['person_bbox'],
                min_confidence=0.5 if FACE_CONFIRM_MIN==0.0 else FACE_CONFIRM_MIN
            )
            if ok_face and face_bbox:
                face_dets.append({
                    'person_bbox': pd['person_bbox'],
                    'face_bbox':   face_bbox,
                    'conf':        pd['conf'],
                })

        tracks, expired_log = manager.update(frame, face_dets)

        # — Log ke terminal
        ev_logger.update(tracks, expired_log)

        draw_tracks(frame, tracks)

        cv2.putText(frame,
                    f"Wajah aktif: {len(tracks)}",
                    (10,26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,(255,255,255),2,cv2.LINE_AA)

        cv2.imshow("Face Tracker — RT-DETR", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('r'):
            manager   = IDManager()
            ev_logger = EventLogger()
            log_separator()
            cprint("  ↺  Semua track di-reset.", 'yellow')
            log_separator()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()