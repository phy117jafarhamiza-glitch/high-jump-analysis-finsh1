import base64
import json
import os
import tempfile

import cv2
import pandas as pd
import streamlit as st

import pose_analysis as pa

# ----------------------------------------------------------------------------
# إعدادات الصفحة
# ----------------------------------------------------------------------------
st.set_page_config(page_title="المحلل الذكي للوثب العالي", page_icon="🧠", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] {direction: rtl; text-align: right;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 المحلل الذكي للوثب العالي (AI Coach)")
st.markdown(
    "يقيس النظام حركة اللاعب بشبكة عصبية لتقدير وضع الجسم (MediaPipe Pose)، "
    "ثم يكتب **Claude** تقرير المدرب وبرنامج التمارين التصحيحية بناءً على القياسات."
)
st.markdown("---")

CLAUDE_MODELS = {
    "Claude Sonnet 5 (موصى به)": "claude-sonnet-5",
    "Claude Opus 5.5 (أدق وأغلى)": "claude-opus-5-5",
    "Claude Haiku 4.5 (أسرع وأرخص)": "claude-haiku-4-5-20251001",
}

# مفتاح Claude: من أسرار Streamlit إن وُجد، وإلا من خانة الإدخال
secret_key = ""
try:
    secret_key = st.secrets.get("ANTHROPIC_API_KEY", "")
except Exception:
    pass

c_key, c_model = st.columns([2, 1])
with c_key:
    if secret_key:
        st.success("🔑 مفتاح Claude محفوظ في إعدادات التطبيق.")
        api_key = secret_key
    else:
        api_key = st.text_input("🔑 مفتاح Claude API (من console.anthropic.com):", type="password")
with c_model:
    model_label = st.selectbox("نموذج المدرب:", list(CLAUDE_MODELS))
use_claude = st.checkbox("كتابة تقرير المدرب بواسطة Claude", value=True,
                         help="بدون هذا الخيار يعرض التطبيق القياسات والمؤشرات فقط (مجاناً).")

st.markdown("---")
col1, col2 = st.columns([1, 1])
with col1:
    st.header("📋 بيانات اللاعب")
    gender = st.selectbox("الجنس:", ["ذكر", "أنثى"])
    height = st.number_input("الطول الكلي (سم):", min_value=140, max_value=240, value=185)
    weight = st.number_input("الوزن (كجم):", min_value=40, max_value=120, value=75)
    best = st.text_input("أفضل إنجاز (اختياري، مثل 1.90 م):", "")
    slowmo = st.selectbox(
        "هل الفيديو مصوّر بالحركة البطيئة؟",
        [1, 2, 4, 8],
        format_func=lambda x: "لا (سرعة عادية)" if x == 1 else f"نعم، أبطأ {x} مرات",
        help="مهم لحساب السرعات والأزمنة بشكل صحيح.",
    )

with col2:
    st.header("🎥 رفع الفيديو للتحليل")
    uploaded_video = st.file_uploader(
        "ارفع فيديو القفزة (كاميرا ثابتة من الجانب، يظهر فيها آخر 3 خطوات والارتقاء والطيران)",
        type=["mp4", "mov", "avi", "m4v"],
    )
    if uploaded_video:
        st.video(uploaded_video)

LABELS = {
    "approach_speed_mps": ("سرعة الاقتراب النهائية", "م/ث"),
    "com_lowering_cm": ("خفض مركز الثقل قبل الارتقاء", "سم"),
    "knee_at_plant_deg": ("زاوية ركبة الارتقاء لحظة وضع القدم", "°"),
    "knee_min_deg": ("أقل زاوية لركبة الارتقاء أثناء الارتكاز", "°"),
    "knee_at_takeoff_deg": ("زاوية ركبة الارتقاء لحظة ترك الأرض", "°"),
    "contact_time_s": ("زمن الارتقاء (تقريبي)", "ث"),
    "trunk_lean_plant_deg": ("ميل الجذع لحظة وضع القدم (+ أمام / − خلف)", "°"),
    "trunk_lean_takeoff_deg": ("ميل الجذع لحظة الارتقاء (+ أمام / − خلف)", "°"),
    "free_knee_lift_pct": ("ارتفاع ركبة الرجل الحرة عن الورك (% من الطول)", "%"),
    "free_knee_angle_deg": ("زاوية ركبة الرجل الحرة لحظة الارتقاء", "°"),
    "arms_raised_count": ("عدد الذراعين المرفوعتين فوق الكتف لحظة الارتقاء", ""),
    "takeoff_angle_deg": ("زاوية الانطلاق", "°"),
    "hip_rise_cm": ("ارتفاع الورك من الارتقاء إلى القمة", "سم"),
    "time_to_peak_s": ("الزمن من الارتقاء إلى القمة", "ث"),
    "hip_angle_at_peak_deg": ("زاوية الورك عند القمة (تقوّس الجسم)", "°"),
    "knee_angle_at_peak_deg": ("زاوية الركبتين عند القمة", "°"),
}
EVENT_NAMES = {
    "approach": "نهاية الاقتراب",
    "plant": "وضع قدم الارتقاء",
    "takeoff": "ترك الأرض",
    "peak": "أعلى نقطة",
    "clearance": "اجتياز العارضة",
}


@st.cache_resource(show_spinner="تحميل نموذج تقدير وضع الجسم (مرة واحدة فقط)...")
def get_model_path():
    return pa.download_model()


def jpeg_b64(rgb):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def build_prompt(result):
    ref = dict(pa.REFERENCE)
    ref["approach_speed_mps"] = pa.REFERENCE["approach_speed_mps"].get(gender)
    metrics_ar = {LABELS[k][0] + (f" ({LABELS[k][1]})" if LABELS[k][1] else ""): v
                  for k, v in result["metrics"].items() if k in LABELS}
    return f"""أنت مدرب وخبير بيوميكانيك متخصص في الوثب العالي بطريقة فوسبري فلوب، وتكتب لمدرب ولاعب في العراق.

بيانات اللاعب:
- الجنس: {gender}
- الطول: {height} سم
- الوزن: {weight} كجم
- أفضل إنجاز: {best or "غير مذكور"}
- رجل الارتقاء: {result["take_leg"]}

القياسات المستخرجة آلياً من الفيديو (MediaPipe Pose، كاميرا واحدة ثنائية الأبعاد):
{json.dumps(metrics_ar, ensure_ascii=False, indent=1)}

أزمنة الأحداث (ث): {json.dumps({EVENT_NAMES[k]: v for k, v in result["event_times"].items()}, ensure_ascii=False)}
نسبة الإطارات التي اكتُشف فيها جسم اللاعب: {result["detection_rate"]}%

قيم مرجعية تقريبية من الأدبيات للاسترشاد (ليست معايير قطعية):
{json.dumps(ref, ensure_ascii=False)}

مؤشرات أولية آلية (قد تكون خاطئة، تحقق منها بالصور):
{json.dumps(result["flags"], ensure_ascii=False)}

مرفق صور للإطارات المفتاحية بالترتيب مع الهيكل المرسوم فوق اللاعب.

حدود القياس التي يجب أن تراعيها:
- التصوير ثنائي الأبعاد، والاقتراب في فوسبري فلوب منحنٍ، فالسرعة والزوايا تقريبية وتتأثر بزاوية الكاميرا.
- ميل الجسم الداخلي نحو مركز المنحنى لا يظهر بوضوح من الجانب.
- إن رأيت في الصور أن اكتشاف المفاصل خاطئ أو أن التصوير غير مناسب فقل ذلك صراحة ولا تبنِ عليه.

المطلوب تقرير باللغة العربية الفصحى بأسلوب مدرب محترف، بالعناوين التالية:
## 1. ملخص سريع
ثلاث جمل: أبرز نقطة قوة، الخطأ الرئيسي، والأولوية التدريبية.
## 2. تحليل مرحلة الاقتراب
## 3. تحليل مرحلة الارتقاء
## 4. تحليل مرحلة الطيران واجتياز العارضة
(في كل مرحلة: ما تُظهره الأرقام والصور مقارنةً بالمرجع، وما هو جيد، وما يحتاج تصحيحاً.)
## 5. الخطأ الميكانيكي الرئيسي
الخطأ، سببه المحتمل، وأثره على الإنجاز.
## 6. البرنامج التصحيحي
برنامج 6 أسابيع مقسّم إلى مراحل، وفي كل تمرين: الاسم، الهدف، الشدة أو الحجم (مجموعات × تكرارات)، ونقطة تعليمية واحدة. ضمّن تمارين تكنيكية (drills)، وتمارين قوة، وتمارين بلايومترية مناسبة للخطأ.
## 7. مؤشرات المتابعة
أي القياسات يجب أن تتحسن وإلى أي قيمة تقريبية عند إعادة التصوير.
## 8. ملاحظات على جودة التصوير
نصائح لتحسين الفيديو القادم إن لزم.

كن محدداً واستند إلى الأرقام. لا تخترع قياسات غير موجودة.
"""


def claude_error_text(e):
    msg = str(e)
    if "credit balance" in msg.lower():
        return "💳 رصيد حساب Anthropic غير كافٍ. اشحن رصيداً من console.anthropic.com ← Billing."
    if "authentication" in msg.lower() or "invalid x-api-key" in msg.lower() or "401" in msg:
        return "🔑 مفتاح Claude غير صالح. تأكد من نسخه كاملاً (يبدأ بـ sk-ant-)."
    if "403" in msg or "permission" in msg.lower():
        return f"⛔ رُفض الطلب من Anthropic (403). التفاصيل: {msg}"
    if "429" in msg or "rate_limit" in msg.lower():
        return "⏳ طلبات كثيرة خلال وقت قصير. انتظر دقيقة ثم أعد المحاولة."
    if "529" in msg or "overloaded" in msg.lower():
        return "⏳ خوادم Claude مشغولة حالياً. أعد المحاولة بعد قليل."
    if "not_found" in msg.lower() or "404" in msg:
        return f"⚠️ النموذج المختار غير متاح لحسابك. جرّب نموذجاً آخر من القائمة. التفاصيل: {msg}"
    return f"حدث خطأ أثناء الاتصال بـ Claude: {msg}"


st.markdown("---")
if st.button("🚀 بدء التحليل", width="stretch"):
    if not uploaded_video:
        st.error("⚠️ يرجى رفع مقطع فيديو للتحليل.")
        st.stop()
    if use_claude and not api_key:
        st.error("⚠️ أدخل مفتاح Claude، أو ألغِ خيار كتابة التقرير لعرض القياسات فقط.")
        st.stop()

    suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded_video.getvalue())
        video_path = f.name

    try:
        # ---------------- المرحلة 1: القياس ----------------
        try:
            model_path = get_model_path()
        except Exception as e:
            raise RuntimeError(f"تعذّر تحميل نموذج تقدير الوضع: {e}")

        bar = st.progress(0.0, text="📐 استخراج مفاصل الجسم من الفيديو...")
        landmarker = pa.create_landmarker(model_path)
        try:
            data = pa.extract_landmarks(
                video_path, landmarker, slowmo_factor=slowmo,
                progress=lambda x: bar.progress(x, text=f"📐 استخراج مفاصل الجسم... {int(x * 100)}%"),
            )
        finally:
            landmarker.close()
        bar.progress(1.0, text="📐 تم استخراج المفاصل. جارٍ حساب المؤشرات...")

        result = pa.analyze(data, height_cm=height, gender=gender)
        frames = pa.key_frames(video_path, data["frame_idx"], result["points"], result["events"])
        bar.empty()

        st.header("📐 القياسات البيوميكانيكية")
        if result["detection_rate"] < 60:
            st.warning(f"اكتُشف جسم اللاعب في {result['detection_rate']}% فقط من الإطارات، فالقياسات أقل دقة.")
        st.caption(f"رجل الارتقاء المكتشفة: {result['take_leg']}. القيم تقريبية لأن التصوير بكاميرا واحدة.")

        rows = [{"المؤشر": LABELS[k][0], "القيمة": v, "الوحدة": LABELS[k][1]}
                for k, v in result["metrics"].items() if k in LABELS]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        if result["flags"]:
            st.subheader("⚠️ مؤشرات أولية")
            for fl in result["flags"]:
                st.markdown(f"- {fl}")

        st.subheader("🖼️ اللحظات المفتاحية")
        names = [n for n in EVENT_NAMES if n in frames]
        cols = st.columns(len(names)) if names else []
        for c, n in zip(cols, names):
            c.image(frames[n], caption=f"{EVENT_NAMES[n]} ({result['event_times'][n]} ث)", width="stretch")

        s = result["series"]
        chart = pd.DataFrame({
            "الزمن (ث)": s["t"],
            "زاوية ركبة الارتقاء (°)": s["knee_take"],
            "ارتفاع الورك (سم)": s["hip_height_cm"],
        }).set_index("الزمن (ث)")
        g1, g2 = st.columns(2)
        g1.markdown("**زاوية ركبة الارتقاء عبر الزمن**")
        g1.line_chart(chart[["زاوية ركبة الارتقاء (°)"]])
        g2.markdown("**ارتفاع الورك (مسار مركز الثقل التقريبي)**")
        g2.line_chart(chart[["ارتفاع الورك (سم)"]])

        csv = pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ تحميل القياسات (CSV)", csv, "high_jump_metrics.csv", "text/csv")

        # ---------------- المرحلة 2: تقرير المدرب ----------------
        if use_claude:
            import anthropic

            st.markdown("---")
            st.header("🏅 تقرير المدرب الذكي")
            content = []
            for n in names:
                content.append({"type": "text", "text": f"الإطار: {EVENT_NAMES[n]}"})
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": jpeg_b64(frames[n])}})
            content.append({"type": "text", "text": build_prompt(result)})

            try:
                client = anthropic.Anthropic(api_key=api_key.strip())
                with client.messages.stream(
                    model=CLAUDE_MODELS[model_label],
                    max_tokens=8000,
                    messages=[{"role": "user", "content": content}],
                ) as stream:
                    report = st.write_stream(stream.text_stream)
                st.caption(f"النموذج: {CLAUDE_MODELS[model_label]}")
                st.download_button("⬇️ تحميل التقرير", str(report).encode("utf-8"),
                                   "high_jump_report.md", "text/markdown")
            except Exception as e:
                st.error(claude_error_text(e))

    except RuntimeError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"حدث خطأ غير متوقع أثناء التحليل: {e}")
    finally:
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass
