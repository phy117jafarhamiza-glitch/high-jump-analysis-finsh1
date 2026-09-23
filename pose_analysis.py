"""
التحليل البيوميكانيكي للوثب العالي (فوسبري فلوب) باستخدام MediaPipe Pose.

المراحل:
1. استخراج 33 نقطة مفصلية من كل إطار (مع تتبع اللاعب وقصّ الصورة حوله لأنه يظهر صغيراً غالباً).
2. تحديد اللحظات المهمة: وضع قدم الارتقاء، ترك الأرض، أعلى نقطة.
3. حساب المؤشرات: سرعة الاقتراب، خفض مركز الثقل، زوايا الركبة، ميل الجذع، زاوية الانطلاق...
"""

import math
import os
import tempfile
import urllib.request

import cv2
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

# أرقام نقاط MediaPipe
NOSE = 0
L_SH, R_SH = 11, 12
L_EL, R_EL = 13, 14
L_WR, R_WR = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANK, R_ANK = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32

SKELETON = [
    (L_SH, R_SH), (L_SH, L_EL), (L_EL, L_WR), (R_SH, R_EL), (R_EL, R_WR),
    (L_SH, L_HIP), (R_SH, R_HIP), (L_HIP, R_HIP),
    (L_HIP, L_KNEE), (L_KNEE, L_ANK), (L_ANK, L_HEEL), (L_HEEL, L_FOOT), (L_ANK, L_FOOT),
    (R_HIP, R_KNEE), (R_KNEE, R_ANK), (R_ANK, R_HEEL), (R_HEEL, R_FOOT), (R_ANK, R_FOOT),
]

# قيم مرجعية تقريبية من أدبيات فوسبري فلوب (Dapena وآخرون) — للاسترشاد وليست معايير قطعية
REFERENCE = {
    "approach_speed_mps": {"ذكر": (6.5, 8.0), "أنثى": (6.0, 7.2)},
    "knee_at_plant_deg": (155, 175),
    "knee_min_deg": (135, 155),
    "com_lowering_cm": (4, 12),
    "takeoff_angle_deg": (40, 55),
    "contact_time_s": (0.14, 0.22),
}


# ----------------------------------------------------------------------------
# 1) النموذج
# ----------------------------------------------------------------------------
def download_model(dest_dir=None):
    dest_dir = dest_dir or tempfile.gettempdir()
    path = os.path.join(dest_dir, "pose_landmarker_full.task")
    if not os.path.exists(path) or os.path.getsize(path) < 1_000_000:
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def create_landmarker(model_path):
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
    )
    return vision.PoseLandmarker.create_from_options(options)


# ----------------------------------------------------------------------------
# 2) استخراج النقاط من الفيديو مع تتبع اللاعب
# ----------------------------------------------------------------------------
def _detect(landmarker, rgb):
    import mediapipe as mp

    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    res = landmarker.detect(img)
    if not res.pose_landmarks:
        return None
    lms = res.pose_landmarks[0]
    arr = np.array([[lm.x, lm.y, getattr(lm, "visibility", 1.0) or 0.0] for lm in lms], dtype=float)
    return arr


def _bbox_from_points(pts, w, h, scale=1.7, min_size=192):
    xs, ys = pts[:, 0], pts[:, 1]
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    size = max(xs.max() - xs.min(), ys.max() - ys.min()) * scale
    size = max(size, min_size)
    x0, y0 = int(max(0, cx - size / 2)), int(max(0, cy - size / 2))
    x1, y1 = int(min(w, cx + size / 2)), int(min(h, cy + size / 2))
    return x0, y0, x1, y1


def _detect_in_region(landmarker, rgb, box):
    x0, y0, x1, y1 = box
    if x1 - x0 < 32 or y1 - y0 < 32:
        return None
    crop = rgb[y0:y1, x0:x1]
    res = _detect(landmarker, crop)
    if res is None:
        return None
    out = res.copy()
    out[:, 0] = x0 + res[:, 0] * (x1 - x0)
    out[:, 1] = y0 + res[:, 1] * (y1 - y0)
    return out


def _tiles(w, h):
    """مناطق متداخلة للبحث عن لاعب صغير داخل إطار واسع."""
    boxes = [(0, 0, w, h)]
    for rows, cols in [(2, 2), (2, 3)]:
        tw, th = w / cols, h / rows
        for r in range(rows):
            for c in range(cols):
                x0 = int(max(0, c * tw - tw * 0.25)); x1 = int(min(w, (c + 1) * tw + tw * 0.25))
                y0 = int(max(0, r * th - th * 0.25)); y1 = int(min(h, (r + 1) * th + th * 0.25))
                boxes.append((x0, y0, x1, y1))
    return boxes


def extract_landmarks(video_path, landmarker, slowmo_factor=1.0, max_frames=600, progress=None):
    """
    يعيد قاموساً فيه:
      points: مصفوفة (N, 33, 3) بإحداثيات البكسل والرؤية، NaN عند عدم الاكتشاف
      t: الزمن الحقيقي لكل إطار بالثواني (بعد تصحيح الحركة البطيئة)
      frame_idx: رقم الإطار في الفيديو
      fps_eff, width, height
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("تعذّر فتح ملف الفيديو.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, math.ceil(total / max_frames)) if total else 1

    points, times, idxs = [], [], []
    prev = None
    lost = 0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step:
            i += 1
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pts = None
        if prev is not None:
            pts = _detect_in_region(landmarker, rgb, _bbox_from_points(prev[:, :2], w, h))
        if pts is None and (prev is None or lost % 3 == 0):
            # بحث كامل: الإطار كله ثم مناطق مكبّرة
            for box in _tiles(w, h):
                pts = _detect_in_region(landmarker, rgb, box)
                if pts is not None and np.nanmean(pts[:, 2]) > 0.35:
                    break
                pts = None
        if pts is not None and np.nanmean(pts[:, 2]) < 0.25:
            pts = None

        if pts is None:
            lost += 1
            if lost > 10:
                prev = None
            points.append(np.full((33, 3), np.nan))
        else:
            lost = 0
            prev = pts
            points.append(pts)
        times.append(i / fps / slowmo_factor)
        idxs.append(i)
        if progress and total:
            progress(min(1.0, i / total))
        i += 1
    cap.release()

    if not points:
        raise RuntimeError("الفيديو لا يحتوي على إطارات قابلة للقراءة.")
    return {
        "points": np.array(points),
        "t": np.array(times),
        "frame_idx": np.array(idxs),
        "fps_eff": fps / step,
        "real_fps": fps / step * slowmo_factor,  # إطارات لكل ثانية حقيقية
        "width": w,
        "height": h,
    }


# ----------------------------------------------------------------------------
# 3) تنظيف السلاسل الزمنية
# ----------------------------------------------------------------------------
def _interp_nans(y, max_gap):
    y = y.copy()
    n = len(y)
    good = ~np.isnan(y)
    if good.sum() < 2:
        return y
    idx = np.arange(n)
    filled = np.interp(idx, idx[good], y[good])
    # لا نملأ الفجوات الطويلة أو الأطراف
    gap_start = None
    for k in range(n):
        if not good[k] and gap_start is None:
            gap_start = k
        if (good[k] or k == n - 1) and gap_start is not None:
            end = k if good[k] else n
            if gap_start > 0 and good[k] and (end - gap_start) <= max_gap:
                y[gap_start:end] = filled[gap_start:end]
            gap_start = None
    return y


def _smooth(y, win):
    if win < 3:
        return y
    out = y.copy()
    half = win // 2
    for k in range(len(y)):
        seg = y[max(0, k - half): k + half + 1]
        if not np.all(np.isnan(seg)) and not np.isnan(y[k]):
            out[k] = np.nanmean(seg)
    return out


def clean_points(data):
    pts = data["points"].copy()
    fps = data["real_fps"]
    max_gap = max(2, int(round(fps * 0.3)))
    win = max(3, int(round(fps * 0.08)) | 1)
    for j in range(33):
        for c in range(2):
            pts[:, j, c] = _smooth(_interp_nans(pts[:, j, c], max_gap), win)
    return pts


# ----------------------------------------------------------------------------
# 4) الحسابات الهندسية
# ----------------------------------------------------------------------------
def angle3(a, b, c):
    """الزاوية عند النقطة b بين b→a و b→c بالدرجات."""
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    cos = np.sum(v1 * v2, axis=-1) / np.where(n == 0, np.nan, n)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _mid(p, a, b):
    return (p[:, a, :2] + p[:, b, :2]) / 2


def _nan_idx(arr, fn):
    if np.all(np.isnan(arr)):
        return None
    return int(fn(arr))


def _r(x, nd=1):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


def analyze(data, height_cm, gender="ذكر"):
    p = clean_points(data)
    t = data["t"]
    n = len(t)
    fps = data["real_fps"]

    hip = _mid(p, L_HIP, R_HIP)
    sh = _mid(p, L_SH, R_SH)
    heel = _mid(p, L_HEEL, R_HEEL)
    detected = ~np.isnan(hip[:, 0])
    raw_hip = (data["points"][:, L_HIP, 0] + data["points"][:, R_HIP, 0]) / 2
    quality = float((~np.isnan(raw_hip)).mean())

    if detected.sum() < max(10, fps * 0.8):
        raise RuntimeError(
            "لم يتمكن النظام من اكتشاف جسم اللاعب في عدد كافٍ من الإطارات. "
            "جرّب فيديو يظهر فيه اللاعب أكبر وأوضح، بكاميرا ثابتة من الجانب."
        )

    knee_L = angle3(p[:, L_HIP, :2], p[:, L_KNEE, :2], p[:, L_ANK, :2])
    knee_R = angle3(p[:, R_HIP, :2], p[:, R_KNEE, :2], p[:, R_ANK, :2])

    # --- أعلى نقطة: أعلى موضع للورك (أصغر y) ---
    peak = _nan_idx(hip[:, 1], np.nanargmin)

    # --- لحظة وضع قدم الارتقاء: أخفض موضع للورك خلال 0.7 ث قبل القمة ---
    w0 = int(np.searchsorted(t, t[peak] - 0.7))
    seg = hip[w0: peak + 1, 1]
    plant = (w0 + _nan_idx(seg, np.nanargmax)) if not np.all(np.isnan(seg)) else max(0, peak - int(fps * 0.3))

    # --- طول الجسم بالبكسل (من مرحلة الاقتراب) لتحويل البكسل إلى سنتيمتر ---
    stature = np.linalg.norm(p[:, NOSE, :2] - heel, axis=1) / 0.93
    appr_mask = detected & (t < t[plant] - 0.15)
    body_px = np.nanpercentile(stature[appr_mask], 75) if appr_mask.sum() >= 3 else np.nanpercentile(stature, 75)
    m_per_px = (height_cm / 100.0) / body_px

    # --- رجل الارتقاء: القدم الأقرب للأرض لحظة الوضع ---
    if np.nan_to_num(p[plant, L_ANK, 1]) >= np.nan_to_num(p[plant, R_ANK, 1]):
        take_leg, free_leg = "L", "R"
        tk_hip, tk_knee, tk_ank = L_HIP, L_KNEE, L_ANK
        fr_hip, fr_knee = R_HIP, R_KNEE
        knee_take = knee_L
    else:
        take_leg, free_leg = "R", "L"
        tk_hip, tk_knee, tk_ank = R_HIP, R_KNEE, R_ANK
        fr_hip, fr_knee = L_HIP, L_KNEE
        knee_take = knee_R

    # --- لحظة ترك الأرض: ارتفاع كاحل الارتقاء بوضوح عن مستواه وقت الوضع ---
    ground_y = p[plant, tk_ank, 1]
    takeoff = None
    for k in range(plant + 1, peak + 1):
        if not np.isnan(p[k, tk_ank, 1]) and p[k, tk_ank, 1] < ground_y - 0.04 * body_px:
            takeoff = k
            break
    if takeoff is None:
        takeoff = min(peak, plant + max(1, int(round(fps * 0.18))))

    # --- اتجاه الركض ---
    back = int(np.searchsorted(t, t[plant] - 0.5))
    direction = np.sign(np.nan_to_num(hip[plant, 0] - hip[back, 0])) or 1.0

    m = {}
    # سرعة الاقتراب الأفقية في آخر 0.5 ث
    dt = t[plant] - t[back]
    if dt > 0.1 and not np.isnan(hip[back, 0]):
        m["approach_speed_mps"] = abs(hip[plant, 0] - hip[back, 0]) * m_per_px / dt
    # خفض مركز الثقل قبل الارتقاء
    a0, a1 = np.searchsorted(t, t[plant] - 0.8), np.searchsorted(t, t[plant] - 0.3)
    if a1 > a0:
        prev_h = np.nanmean(hip[a0:a1, 1])
        m["com_lowering_cm"] = (hip[plant, 1] - prev_h) * m_per_px * 100
    # زوايا ركبة الارتقاء
    m["knee_at_plant_deg"] = knee_take[plant]
    m["knee_min_deg"] = np.nanmin(knee_take[plant: takeoff + 1])
    m["knee_at_takeoff_deg"] = knee_take[takeoff]
    # زمن الارتقاء التقريبي
    m["contact_time_s"] = t[takeoff] - t[plant]
    # ميل الجذع: موجب = للأمام باتجاه الركض، سالب = للخلف
    def trunk(k):
        v = sh[k] - hip[k]
        return math.degrees(math.atan2(v[0] * direction, -v[1]))
    m["trunk_lean_plant_deg"] = trunk(plant)
    m["trunk_lean_takeoff_deg"] = trunk(takeoff)
    # الرجل الحرة: ارتفاع الركبة نسبة للورك (موجب = الركبة أعلى من الورك)
    m["free_knee_lift_pct"] = (p[takeoff, fr_hip, 1] - p[takeoff, fr_knee, 1]) / body_px * 100
    m["free_knee_angle_deg"] = angle3(p[takeoff, fr_hip, :2], p[takeoff, fr_knee, :2],
                                      p[takeoff, L_ANK if free_leg == "L" else R_ANK, :2])
    # الذراعان: الرسغان فوق الكتفين؟
    wr_up = [(p[takeoff, wr, 1] < p[takeoff, s, 1]) for wr, s in [(L_WR, L_SH), (R_WR, R_SH)]]
    m["arms_raised_count"] = int(sum(bool(x) for x in wr_up))
    # زاوية الانطلاق من سرعة الورك بعد ترك الأرض
    k2 = min(peak, int(np.searchsorted(t, t[takeoff] + 0.1)))
    if k2 > takeoff:
        dx = abs(hip[k2, 0] - hip[takeoff, 0])
        dy = hip[takeoff, 1] - hip[k2, 1]
        m["takeoff_angle_deg"] = math.degrees(math.atan2(dy, dx))
    # الطيران
    m["hip_rise_cm"] = (hip[takeoff, 1] - hip[peak, 1]) * m_per_px * 100
    m["time_to_peak_s"] = t[peak] - t[takeoff]
    arch = angle3(sh[peak], hip[peak], _mid(p, L_KNEE, R_KNEE)[peak])
    m["hip_angle_at_peak_deg"] = float(arch)
    m["knee_angle_at_peak_deg"] = float(np.nanmean([knee_L[peak], knee_R[peak]]))

    metrics = {k: _r(v) for k, v in m.items()}
    metrics["contact_time_s"] = _r(m["contact_time_s"], 2)
    metrics["time_to_peak_s"] = _r(m["time_to_peak_s"], 2)
    metrics["arms_raised_count"] = m["arms_raised_count"]

    events = {
        "approach": max(0, int(np.searchsorted(t, t[plant] - 0.5))),
        "plant": int(plant),
        "takeoff": int(takeoff),
        "peak": int(peak),
        "clearance": min(n - 1, int(np.searchsorted(t, t[peak] + 0.2))),
    }

    ground_hip = np.nanmax(hip[:, 1])
    series = {
        "t": t,
        "knee_take": knee_take,
        "hip_height_cm": (ground_hip - hip[:, 1]) * m_per_px * 100,
    }

    return {
        "metrics": metrics,
        "events": events,
        "event_times": {k: round(float(t[v]), 2) for k, v in events.items()},
        "take_leg": "اليسرى" if take_leg == "L" else "اليمنى",
        "detection_rate": round(quality * 100),
        "series": series,
        "points": p,
        "flags": build_flags(metrics, gender),
    }


# ----------------------------------------------------------------------------
# 5) مؤشرات أولية مبنية على قواعد
# ----------------------------------------------------------------------------
def build_flags(m, gender):
    flags = []
    lo_speed = REFERENCE["approach_speed_mps"].get(gender, (6.0, 7.5))[0]
    v = m.get("approach_speed_mps")
    if v is not None and v < lo_speed - 0.8:
        flags.append(f"سرعة الاقتراب النهائية منخفضة ({v} م/ث).")
    v = m.get("com_lowering_cm")
    if v is not None and v < 2:
        flags.append("لا يظهر خفض واضح لمركز الثقل في الخطوات الأخيرة قبل الارتقاء.")
    v = m.get("knee_at_plant_deg")
    if v is not None and v < 150:
        flags.append(f"ركبة الارتقاء مثنية كثيراً لحظة وضع القدم ({v}°).")
    v = m.get("knee_min_deg")
    if v is not None and v < 130:
        flags.append(f"انثناء مفرط لركبة الارتقاء أثناء الارتكاز ({v}°) يسبب فقداناً للطاقة.")
    v = m.get("trunk_lean_takeoff_deg")
    if v is not None and v > 12:
        flags.append(f"ميل الجذع للأمام لحظة الارتقاء ({v}°).")
    v = m.get("free_knee_lift_pct")
    if v is not None and v < -8:
        flags.append("مرجحة الرجل الحرة ضعيفة (الركبة منخفضة لحظة الارتقاء).")
    if m.get("arms_raised_count") == 0:
        flags.append("لا تُستخدم الذراعان في المرجحة للأعلى لحظة الارتقاء.")
    v = m.get("takeoff_angle_deg")
    if v is not None and v < 38:
        flags.append(f"زاوية الانطلاق منخفضة ({v}°): تحويل ضعيف للسرعة الأفقية إلى عمودية.")
    elif v is not None and v > 65:
        flags.append(f"زاوية الانطلاق عالية جداً ({v}°): فقدان للسرعة الأفقية.")
    v = m.get("contact_time_s")
    if v is not None and v > 0.26:
        flags.append(f"زمن الارتقاء طويل ({v} ث).")
    return flags


# ----------------------------------------------------------------------------
# 6) الإطارات المفتاحية مع رسم الهيكل
# ----------------------------------------------------------------------------
def draw_skeleton(frame_bgr, pts, label=None):
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    th = max(2, int(round(min(w, h) / 300)))
    for a, b in SKELETON:
        pa, pb = pts[a, :2], pts[b, :2]
        if np.any(np.isnan(pa)) or np.any(np.isnan(pb)):
            continue
        cv2.line(img, tuple(int(v) for v in pa), tuple(int(v) for v in pb), (0, 220, 255), th, cv2.LINE_AA)
    for j in range(11, 33):
        if not np.any(np.isnan(pts[j, :2])):
            cv2.circle(img, tuple(int(v) for v in pts[j, :2]), th + 1, (255, 80, 40), -1, cv2.LINE_AA)
    return img


def crop_around(img, pts, scale=1.8, min_size=240):
    h, w = img.shape[:2]
    valid = pts[~np.isnan(pts[:, 0])]
    if len(valid) < 5:
        return img
    x0, y0, x1, y1 = _bbox_from_points(valid[:, :2], w, h, scale=scale, min_size=min_size)
    return img[y0:y1, x0:x1]


def key_frames(video_path, frame_indices, points, events, max_side=720):
    """يعيد قاموساً {اسم الحدث: صورة RGB مقصوصة حول اللاعب مع الهيكل}."""
    wanted = {}
    for name, i in events.items():
        wanted.setdefault(int(frame_indices[i]), []).append((name, i))
    out = {}
    cap = cv2.VideoCapture(video_path)
    k = 0
    last = max(wanted) if wanted else -1
    while k <= last:
        ok, frame = cap.read()
        if not ok:
            break
        for name, i in wanted.get(k, []):
            img = crop_around(draw_skeleton(frame, points[i]), points[i])
            hh, ww = img.shape[:2]
            s = max_side / max(hh, ww)
            if s < 1:
                img = cv2.resize(img, (int(ww * s), int(hh * s)), interpolation=cv2.INTER_AREA)
            out[name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        k += 1
    cap.release()
    return out
