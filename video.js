// 영상 행동분석 (실험적): COCO-SSD로 고양이 탐지 + 박스 추적 휴리스틱
// 정밀 자세(꼬리/귀) 인식이 아니라, 움직임·자세비율 기반의 "행동 추정"입니다.

const fileInput = document.getElementById("videoFile");
const videoEl = document.getElementById("videoEl");
const analyzeBtn = document.getElementById("analyzeBtn");
const vidStatus = document.getElementById("vidStatus");
const canvas = document.getElementById("vidCanvas");
const ctx = canvas.getContext("2d");

let model = null;

fileInput.onchange = () => {
  const f = fileInput.files[0];
  if (!f) return;
  videoEl.src = URL.createObjectURL(f);
  videoEl.style.display = "block";
  analyzeBtn.disabled = false;
  document.getElementById("vidResult").classList.remove("show");
};

async function loadModel() {
  if (model) return model;
  if (!window.cocoSsd) throw new Error("모델 라이브러리 로드 실패(인터넷 확인)");
  vidStatus.textContent = "AI 모델 로드 중... (최초 1회, 잠시만요)";
  model = await window.cocoSsd.load({ base: "lite_mobilenet_v2" });
  return model;
}

// 특정 시각으로 이동 후 프레임 확보
function seek(t) {
  return new Promise((res) => {
    const on = () => { videoEl.removeEventListener("seeked", on); res(); };
    videoEl.addEventListener("seeked", on);
    videoEl.currentTime = Math.min(t, videoEl.duration - 0.05);
  });
}

async function detectCat() {
  const w = videoEl.videoWidth, h = videoEl.videoHeight;
  canvas.width = w; canvas.height = h;
  ctx.drawImage(videoEl, 0, 0, w, h);
  const preds = await model.detect(canvas);
  const cats = preds.filter((p) => p.class === "cat");
  if (!cats.length) return null;
  cats.sort((a, b) => b.score - a.score);
  const [x, y, bw, bh] = cats[0].bbox;
  return { cx: (x + bw / 2) / w, cy: (y + bh / 2) / h, w: bw / w, h: bh / h, score: cats[0].score };
}

analyzeBtn.onclick = async () => {
  try {
    analyzeBtn.disabled = true;
    await loadModel();
    if (!videoEl.duration || !isFinite(videoEl.duration)) {
      vidStatus.textContent = "영상을 먼저 불러와 주세요."; analyzeBtn.disabled = false; return;
    }
    vidStatus.textContent = "프레임 분석 중...";

    const dur = videoEl.duration;
    const N = Math.min(16, Math.max(6, Math.floor(dur * 2))); // 초당 ~2프레임, 6~16개
    const frames = [];
    for (let i = 0; i < N; i++) {
      await seek(dur * (i + 0.5) / N);
      frames.push(await detectCat());
      vidStatus.textContent = `프레임 분석 중... ${i + 1}/${N}`;
    }

    const found = frames.filter(Boolean);
    if (found.length < 2) {
      vidStatus.textContent = "";
      showResult({ behavior: "탐지 실패", ko: "고양이를 찾지 못했어요",
        msg: "고양이가 화면에 또렷이 보이는 영상으로 다시 시도해 주세요.", metrics: {}, conf: 0 });
      analyzeBtn.disabled = false;
      return;
    }

    // ---- 휴리스틱 지표 ----
    let motion = 0, mc = 0;
    for (let i = 1; i < frames.length; i++) {
      if (frames[i] && frames[i - 1]) {
        const dx = frames[i].cx - frames[i - 1].cx, dy = frames[i].cy - frames[i - 1].cy;
        motion += Math.hypot(dx, dy); mc++;
      }
    }
    motion = mc ? motion / mc : 0;                          // 평균 이동량(0~1)
    const aspect = avg(found.map((f) => f.w / f.h));        // 가로/세로 비
    const sizeVar = std(found.map((f) => f.w * f.h));       // 접근/후퇴 변동
    const coverage = found.length / frames.length;         // 탐지 안정성

    const beh = classify(motion, aspect, sizeVar);
    beh.metrics = {
      "움직임": clamp(motion * 6),
      "활동 변화": clamp(sizeVar * 8),
      "자세(늘어짐)": clamp((aspect - 0.6) / 1.2),
      "탐지 안정성": coverage,
    };
    beh.conf = coverage;
    vidStatus.textContent = "";
    showResult(beh);
    fuse(beh);
    analyzeBtn.disabled = false;
  } catch (e) {
    vidStatus.textContent = "오류: " + e.message;
    analyzeBtn.disabled = false;
  }
};

function classify(motion, aspect, sizeVar) {
  if (motion > 0.12 || sizeVar > 0.05)
    return { behavior: "active", ko: "활발 · 놀고 싶거나 흥분", msg: "많이 움직여요. 놀이 욕구가 높거나 들떠 있는 상태일 수 있어요." };
  if (aspect > 1.4)
    return { behavior: "relaxed", ko: "편안 · 안정", msg: "옆으로 늘어진 자세로 가만히 있어요. 편안하고 안정된 상태로 보여요." };
  if (aspect < 0.85)
    return { behavior: "alert", ko: "경계 · 긴장", msg: "웅크린 자세예요. 주변을 경계하거나 긴장했을 수 있어요." };
  return { behavior: "explore", ko: "탐색 중 · 차분", msg: "천천히 움직이며 주변을 살피는 차분한 상태로 보여요." };
}

function showResult(b) {
  document.getElementById("vidResult").classList.add("show");
  document.getElementById("vidLabel").textContent =
    b.conf ? `${b.ko} (신뢰도 ${Math.round(b.conf * 100)}%)` : b.ko;
  document.getElementById("vidMsg").textContent = b.msg;
  document.getElementById("vidMetrics").innerHTML = Object.entries(b.metrics || {})
    .map(([k, v]) => `<div>${k} ${Math.round(v * 100)}%</div>
      <div class="bar"><div style="width:${Math.round(v * 100)}%"></div></div>`).join("");
  document.getElementById("vidDisc").textContent =
    "※ 움직임·자세 비율 기반의 실험적 추정입니다. 꼬리·귀 등 정밀 신호는 분석하지 않습니다.";
}

// 오디오 결과와 융합
function fuse(beh) {
  const a = window.lastCatAudio;
  if (!a) return;
  const el = document.getElementById("vidDisc");
  el.textContent += `  🔗 소리(${a.ko}) + 행동(${beh.ko}) → 종합: ${combine(a.ko, beh.behavior)}`;
}
function combine(audioKo, behavior) {
  if (behavior === "active") return "지금 놀아주면 좋아할 것 같아요!";
  if (behavior === "alert") return "조심스러운 상태 — 천천히 다가가세요.";
  if (behavior === "relaxed") return "편안한 시간을 보내는 중이에요.";
  return audioKo + " 신호가 가장 두드러져요.";
}

const avg = (a) => a.reduce((s, v) => s + v, 0) / a.length;
const std = (a) => { const m = avg(a); return Math.sqrt(avg(a.map((v) => (v - m) ** 2))); };
const clamp = (v) => Math.max(0, Math.min(1, v));
