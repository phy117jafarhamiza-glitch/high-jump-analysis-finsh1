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

    points, times, idxs, cam = [], [], [], []
    prev = None
    lost = 0
    i = 0
    # تقدير حركة الكاميرا (تحريك/إمالة) من الخلفية
    sw = 320
    sh_ = max(32, int(round(h * sw / max(w, 1))))
    sx, sy = w / sw, h / sh_
    hann = cv2.createHanningWindow((sw, sh_), cv2.CV_32F)
    prev_small = None
    cum = np.zeros(2)
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

        # حركة الكاميرا: نخفي منطقة اللاعب حتى لا تؤثر حركته على التقدير
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (sw, sh_)).astype(np.float32)
        body = pts if pts is not None else prev
        if body is not None:
            x0, y0, x1, y1 = _bbox_from_points(body[:, :2], w, h, scale=1.3, min_size=0)
            small[int(y0 / sy):int(y1 / sy) + 1, int(x0 / sx):int(x1 / sx) + 1] = float(small.mean())
        if prev_small is not None:
            (dx, dy), resp = cv2.phaseCorrelate(prev_small, small, hann)
            if resp > 0.05 and abs(dx) < sw / 4 and abs(dy) < sh_ / 4:
                cum += (dx * sx, dy * sy)
        prev_small = small

        if pts is None:
            lost += 1
            if lost > 10:
                prev = None
            points.append(np.full((33, 3), np.nan))
        else:
            lost = 0
            prev = pts
            points.append(pts)
        cam.append(cum.copy())
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
        "cam_shift": np.array(cam),  # إزاحة الخلفية التراكمية بالبكسل
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


def _seg_len(p):
    """مجموع أطوال الفخذ والساق والجذع بالبكسل: ثابت مهما انثنى الجسم، فنستخدمه مقياساً لكل إطار."""
    d = lambda a, b: np.linalg.norm(p[:, a, :2] - p[:, b, :2], axis=1)
    thigh = np.nanmean([d(L_HIP, L_KNEE), d(R_HIP, R_KNEE)], axis=0)
    shank = np.nanmean([d(L_KNEE, L_ANK), d(R_KNEE, R_ANK)], axis=0)
    trunk = np.linalg.norm(_mid(p, L_SH, R_SH) - _mid(p, L_HIP, R_HIP), axis=1)
    return thigh + shank + trunk


SEG_RATIO = 0.779   # (الفخذ + الساق + الجذع) / الطول الكلي — نسب Winter التشريحية
LEG_RATIO = 0.491   # (الفخذ + الساق) / الطول الكلي


def reject_outliers(points, fps):
    """حذف الإطارات التي يكون فيها الهيكل مشوّهاً أو قافزاً بشكل غير منطقي (اكتشاف خاطئ)."""
    pts = points.copy()
    seg = _seg_len(pts)
    half = max(2, int(fps * 0.5))
    bad = np.zeros(len(seg), bool)
    for k in range(len(seg)):
        if np.isnan(seg[k]):
            continue
        med = np.nanmedian(seg[max(0, k - half): k + half + 1])
        if med > 0 and not (0.55 < seg[k] / med < 1.6):
            bad[k] = True
    hip = _mid(pts, L_HIP, R_HIP)
    last = None
    for k in range(len(seg)):
        if bad[k] or np.isnan(hip[k, 0]):
            continue
        if last is not None and (k - last) <= 3:
            jump = np.linalg.norm(hip[k] - hip[last]) / (k - last)
            if jump > 0.8 * seg[k]:
                bad[k] = True
                continue
        last = k
    pts[bad] = np.nan
    return pts


def clean_points(points, fps):
    pts = points.copy()
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


def _r(x, nd=1):
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else round(x, nd)


def _runs(mask):
    """قائمة (بداية، نهاية) للمقاطع المتصلة التي قيمتها True."""
    out, start = [], None
    for k, v in enumerate(mask):
        if v and start is None:
            start = k
        if (not v or k == len(mask) - 1) and start is not None:
            out.append((start, k if v else k - 1))
            start = None
    return out


def _at(t, sec):
    return int(np.clip(np.searchsorted(t, sec), 0, len(t) - 1))


def analyze(data, height_cm, gender="ذكر"):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        res = _analyze(data, height_cm, gender)
        if not res.get("slowmo_suggest"):
            return res
        # الزمن غير منطقي: نجرب عوامل الحركة البطيئة الشائعة ونختار الأقرب للفيزياء
        best, best_err = None, None
        for k in (2, 4, 8):
            d = dict(data)
            d["t"] = np.asarray(data["t"]) / k
            d["real_fps"] = data["real_fps"] * k
            try:
                r = _analyze(d, height_cm, gender)
            except Exception:
                continue
            ttp = r["metrics"].get("time_to_peak_s") or 0
            err = abs(ttp - 0.45) + (1.0 if r.get("slowmo_suggest") else 0)
            if best is None or err < best_err:
                best, best_err, best_k = r, err, k
        if best is None:
            return res
        best["notes"] = [n for n in best["notes"] if "الحركة البطيئة" not in n]
        best["notes"].insert(0, (
            f"الأزمنة في الفيديو أطول من الممكن بشرياً، فالأرجح أنه مصوّر بالحركة البطيئة. "
            f"طُبّق تصحيح ×{best_k} تلقائياً. إن كنت تعرف العامل الصحيح فاختره من خانة الحركة البطيئة وأعد التحليل."
        ))
        best["slowmo_applied"] = best_k
        return best


def _analyze(data, height_cm, gender):
    t = data["t"]
    n = len(t)
    fps = data["real_fps"]
    H = height_cm / 100.0

    raw = reject_outliers(data["points"], fps)
    raw_hip = _mid(raw, L_HIP, R_HIP)
    quality = float((~np.isnan(raw_hip[:, 0])).mean())
    if (~np.isnan(raw_hip[:, 0])).sum() < max(10, fps * 0.8):
        raise RuntimeError(
            "لم يتمكن النظام من اكتشاف جسم اللاعب في عدد كافٍ من الإطارات. "
            "جرّب فيديو يظهر فيه اللاعب أكبر وأوضح، بكاميرا ثابتة من الجانب."
        )

    p = clean_points(raw, fps)                       # إحداثيات الصورة (للزوايا والرسم)
    cam = data.get("cam_shift")
    cam = np.zeros((n, 2)) if cam is None else np.asarray(cam, float)
    W = p.copy()                                     # إحداثيات مثبّتة بعد طرح حركة الكاميرا
    W[:, :, 0] -= cam[:, 0:1]
    W[:, :, 1] -= cam[:, 1:2]

    long_win = max(3, int(round(fps * 0.3)) | 1)
    seg = _smooth(_interp_nans(_seg_len(p), max(2, int(fps))), long_win)
    px_per_m = seg / (SEG_RATIO * H)
    leg_px = px_per_m * LEG_RATIO * H

    hip = _mid(p, L_HIP, R_HIP)
    sh = _mid(p, L_SH, R_SH)
    hipW = _mid(W, L_HIP, R_HIP)
    knee = {
        "L": angle3(p[:, L_HIP, :2], p[:, L_KNEE, :2], p[:, L_ANK, :2]),
        "R": angle3(p[:, R_HIP, :2], p[:, R_KNEE, :2], p[:, R_ANK, :2]),
    }
    v = sh - hip
    trunk_abs = np.degrees(np.arctan2(np.abs(v[:, 0]), -v[:, 1]))  # 0 = عمودي، 90 = أفقي
    trunk_s = _smooth(trunk_abs, max(3, int(round(fps * 0.1)) | 1))

    # --- اجتياز العارضة: أول فترة يصبح فيها الجسم شبه أفقي بعد أن كان قائماً ---
    # (نستبعد الهبوط على المرتبة: يأتي بعد الاجتياز وقد يكون الجسم فيه مقلوباً)
    min_len = max(2, int(fps * 0.08))
    trunk_n = np.nan_to_num(trunk_s)
    runs = [r for r in _runs((trunk_n > 55) & (trunk_n < 130)) if r[1] - r[0] + 1 >= min_len]
    look = max(1, int(fps * 1.0))
    jump_runs = [r for r in runs if np.any((trunk_n[max(0, r[0] - look):r[0]] > 0) &
                                           (trunk_n[max(0, r[0] - look):r[0]] < 35))]
    if jump_runs:
        a, b = jump_runs[0]
        hip_run = _mid(W, L_HIP, R_HIP)[a:b + 1, 1]
        clearance = a + (int(np.nanargmin(hip_run)) if not np.all(np.isnan(hip_run))
                         else int(np.nanargmax(trunk_s[a:b + 1])))
    elif runs:
        a, b = max(runs, key=lambda r: r[1] - r[0])
        clearance = a + int(np.nanargmax(trunk_s[a:b + 1]))
    else:
        clearance = int(np.nanargmax(trunk_s))

    # --- آخر لحظة كان فيها الجذع قائماً قبل الاجتياز ---
    upright = [k for k in range(clearance) if not np.isnan(trunk_s[k]) and trunk_s[k] < 35]
    upright_end = upright[-1] if upright else max(0, clearance - int(fps * 0.3))

    # --- الارتكاز الأخير: قدم شبه ثابتة (بعد تعويض حركة الكاميرا) ---
    s0 = _at(t, t[upright_end] - 0.6)
    s1 = min(clearance, _at(t, t[upright_end] + 0.15))
    best = None
    for leg, ank_i, other_i in (("L", L_ANK, R_ANK), ("R", R_ANK, L_ANK)):
        ank = W[:, ank_i, :2]
        vel = np.linalg.norm(np.gradient(ank, axis=0), axis=1) * fps / leg_px
        low = (np.nan_to_num(vel, nan=99) < 2.5)
        low[:s0] = False
        low[s1 + 1:] = False
        for a, b in _runs(low):
            if b - a + 1 < 2:
                continue
            lower = np.nanmean(p[a:b + 1, ank_i, 1] - p[a:b + 1, other_i, 1]) > 0
            if lower and (best is None or b > best[2]):
                best = (leg, a, b)
    if best:
        leg, plant, takeoff = best
        # التنعيم والاشتقاق يقصّران طرفي الارتكاز بإطار تقريباً
        plant, takeoff = max(0, plant - 1), min(n - 1, takeoff + 1)
        if t[takeoff] - t[plant] > 0.4:
            plant = _at(t, t[takeoff] - 0.25)
        detection_method = "stance"
    else:
        # بديل: القدم الأخفض قرب نهاية الوضع القائم
        k = upright_end
        leg = "L" if np.nan_to_num(p[k, L_ANK, 1]) >= np.nan_to_num(p[k, R_ANK, 1]) else "R"
        takeoff = upright_end
        plant = _at(t, t[takeoff] - 0.18)
        detection_method = "fallback"
    takeoff = max(takeoff, plant + 1) if takeoff < n - 1 else takeoff

    free = "R" if leg == "L" else "L"
    I = {"L": (L_HIP, L_KNEE, L_ANK, L_WR, L_SH), "R": (R_HIP, R_KNEE, R_ANK, R_WR, R_SH)}
    tk_hip, tk_knee, tk_ank, _, _ = I[leg]
    fr_hip, fr_knee, fr_ank, _, _ = I[free]
    knee_take = knee[leg]

    # --- أعلى نقطة للورك بعد الارتقاء ---
    e = min(n - 1, _at(t, t[clearance] + 0.3))
    segw = hipW[takeoff:e + 1, 1]
    peak = takeoff + int(np.nanargmin(segw)) if not np.all(np.isnan(segw)) else clearance

    # --- هل الكاميرا ثابتة؟ ---
    w0 = _at(t, t[plant] - 1.0)
    sc = px_per_m[w0:e + 1]
    zoom_ratio = float(np.nanmax(sc) / np.nanmin(sc)) if np.any(~np.isnan(sc)) else 1.0
    zoomed = zoom_ratio > 1.3
    pan_px = float(np.nanmax(np.linalg.norm(cam[w0:e + 1] - cam[w0], axis=1)))
    panned = pan_px > 0.3 * float(np.nanmedian(leg_px[w0:e + 1]))
    notes = []
    if zoomed:
        notes.append(
            f"الكاميرا قرّبت أو بعّدت الصورة (Zoom) أثناء القفزة بنسبة ×{zoom_ratio:.1f}، "
            "لذلك لم تُحسب السرعة وخفض مركز الثقل وزاوية الانطلاق وارتفاع الورك. الزوايا المفصلية ما زالت صالحة."
        )
    elif panned:
        notes.append("الكاميرا تحرّكت مع اللاعب؛ عُوّضت حركتها تقريبياً، فالسرعات والارتفاعات أقل دقة.")
    if detection_method == "fallback":
        notes.append("تعذّر تحديد لحظة الارتكاز بدقة، فحُدّدت تقريبياً من وضع الجذع.")
    absolute_ok = not zoomed

    # --- فحص منطقية الزمن: من ترك الأرض إلى الاجتياز يستغرق عادةً 0.3–0.6 ث ---
    slowmo_suggest = None
    flight = t[clearance] - t[takeoff]
    if flight > 0.9:
        k = flight / 0.45
        slowmo_suggest = min([2, 4, 8], key=lambda c: abs(c - k))
        notes.insert(0, (
            f"الزمن من ترك الأرض إلى الاجتياز {flight:.2f} ث، وهذا أطول من الممكن بشرياً (عادةً 0.3–0.6 ث). "
            f"الأرجح أن الفيديو بالحركة البطيئة (حوالي ×{slowmo_suggest}). اختر ذلك في خانة الحركة البطيئة وأعد التحليل؛ "
            "كل الأزمنة والسرعات الحالية غير صحيحة."
        ))

    def m_between(a, b):
        return (px_per_m[a] + px_per_m[b]) / 2

    back = _at(t, t[plant] - 0.5)
    direction = np.sign(np.nan_to_num(hipW[plant, 0] - hipW[back, 0])) or 1.0

    m = {}
    if absolute_ok:
        dt = t[plant] - t[back]
        if dt > 0.15 and not np.isnan(hipW[back, 0]):
            m["approach_speed_mps"] = abs(hipW[plant, 0] - hipW[back, 0]) / m_between(back, plant) / dt
        a0, a1 = _at(t, t[plant] - 0.8), _at(t, t[plant] - 0.3)
        if a1 > a0:
            m["com_lowering_cm"] = (hipW[plant, 1] - np.nanmean(hipW[a0:a1, 1])) / px_per_m[plant] * 100
    m["knee_at_plant_deg"] = knee_take[plant]
    m["knee_min_deg"] = np.nanmin(knee_take[plant: takeoff + 1])
    m["knee_at_takeoff_deg"] = knee_take[takeoff]
    m["contact_time_s"] = t[takeoff] - t[plant]

    def trunk_signed(k):
        vv = sh[k] - hip[k]
        return math.degrees(math.atan2(vv[0] * direction, -vv[1]))

    m["trunk_lean_plant_deg"] = trunk_signed(plant)
    m["trunk_lean_takeoff_deg"] = trunk_signed(takeoff)
    m["free_knee_lift_pct"] = (p[takeoff, fr_hip, 1] - p[takeoff, fr_knee, 1]) / (px_per_m[takeoff] * H) * 100
    m["free_knee_angle_deg"] = knee[free][takeoff]
    raised = sum(bool(p[takeoff, wr, 1] < p[takeoff, s, 1]) for wr, s in [(L_WR, L_SH), (R_WR, R_SH)])
    if absolute_ok:
        k2 = min(peak, _at(t, t[takeoff] + 0.1))
        if k2 > takeoff:
            dx = abs(hipW[k2, 0] - hipW[takeoff, 0])
            dy = hipW[takeoff, 1] - hipW[k2, 1]
            m["takeoff_angle_deg"] = math.degrees(math.atan2(dy, dx))
        m["hip_rise_cm"] = (hipW[takeoff, 1] - hipW[peak, 1]) / m_between(takeoff, peak) * 100
    m["time_to_peak_s"] = t[peak] - t[takeoff]
    m["hip_angle_at_peak_deg"] = angle3(sh[clearance], hip[clearance], _mid(p, L_KNEE, R_KNEE)[clearance])
    m["knee_angle_at_peak_deg"] = np.nanmean([knee["L"][clearance], knee["R"][clearance]])

    metrics = {k: _r(val) for k, val in m.items()}
    metrics["contact_time_s"] = _r(m["contact_time_s"], 2)
    metrics["time_to_peak_s"] = _r(m["time_to_peak_s"], 2)
    metrics["arms_raised_count"] = int(raised)

    clear_ev = clearance if clearance != peak else min(n - 1, _at(t, t[peak] + 0.2))
    events = {
        "approach": _at(t, t[plant] - 0.5),
        "plant": int(plant),
        "takeoff": int(takeoff),
        "peak": int(peak),
        "clearance": int(clear_ev),
    }

    ground = hipW[plant, 1] + LEG_RATIO * H * px_per_m[plant] * 0.95  # مستوى الأرض التقريبي
    series = {
        "t": t,
        "knee_take": knee_take,
        "trunk": trunk_abs,
        "hip_height_cm": ((ground - hipW[:, 1]) / px_per_m * 100) if absolute_ok else None,
    }

    return {
        "metrics": metrics,
        "events": events,
        "event_times": {k: round(float(t[v]), 2) for k, v in events.items()},
        "take_leg": "اليسرى" if leg == "L" else "اليمنى",
        "detection_rate": round(quality * 100),
        "series": series,
        "points": p,
        "notes": notes,
        "camera": {"zoom_ratio": round(zoom_ratio, 2), "zoomed": zoomed, "panned": panned},
        "slowmo_suggest": slowmo_suggest,
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
    if x1 - x0 < 16 or y1 - y0 < 16:
        return img
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
