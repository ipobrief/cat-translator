// 완전 클라이언트용: 녹음 -> dsp.js 특징 -> model.json RandomForest 추론 (서버 불필요)
import { extractFeatures } from "./lib/dsp.js";

let recording = false, audioCtx, stream, processor, source, chunks = [], sampleRate = 16000;
let MODEL = null, recTimer = null, recStart = 0;

const recBtn = document.getElementById("recBtn");
const catStatus = document.getElementById("catStatus");

// 모델 로드
fetch("./model.json").then((r) => r.json()).then((m) => { MODEL = m; })
  .catch(() => { catStatus.textContent = "모델 로드 실패 (model.json 확인)"; });

recBtn.onclick = async () => { if (!recording) await startRec(); else await stopRec(); };

async function startRec() {
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
  catch { catStatus.textContent = "마이크 권한이 필요합니다."; return; }
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  sampleRate = audioCtx.sampleRate;
  source = audioCtx.createMediaStreamSource(stream);
  processor = audioCtx.createScriptProcessor(4096, 1, 1);
  chunks = [];
  processor.onaudioprocess = (e) => chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  source.connect(processor); processor.connect(audioCtx.destination);
  recording = true;
  recBtn.innerHTML = "■ 녹음 중지"; recBtn.classList.add("rec");
  recStart = Date.now();
  const tick = () => {
    const s = ((Date.now() - recStart) / 1000).toFixed(1);
    catStatus.innerHTML = `<span class="rec-dot"></span>녹음 중 ${s}s · 고양이 소리를 들려주세요`;
  };
  tick(); recTimer = setInterval(tick, 100);
}

async function stopRec() {
  recording = false;
  clearInterval(recTimer);
  recBtn.innerHTML = "녹음 시작"; recBtn.classList.remove("rec");
  processor.disconnect(); source.disconnect();
  stream.getTracks().forEach((t) => t.stop());
  await audioCtx.close();
  if (!MODEL) { catStatus.textContent = "모델을 불러오는 중입니다. 잠시 후 다시 시도해 주세요."; return; }

  catStatus.innerHTML = `<span class="spin"></span>분석 중...`;
  const pcm = flatten(chunks);
  if (pcm.length < sampleRate * 0.2) { catStatus.textContent = "녹음이 너무 짧아요. 다시 시도해 주세요."; return; }
  const feats = extractFeatures(pcm, sampleRate);
  showCat(predict(feats));
  catStatus.textContent = "";
}

// ---- RandomForest 추론 ----
function predict(feats) {
  const { scaler, trees, classes, labels } = MODEL;
  const x = feats.map((v, i) => (v - scaler.mean[i]) / (scaler.scale[i] || 1));
  const prob = new Array(classes.length).fill(0);
  for (const t of trees) {
    let node = 0;
    while (t.cl[node] !== -1) {
      node = x[t.f[node]] <= t.th[node] ? t.cl[node] : t.cr[node];
    }
    const v = t.val[node];
    for (let i = 0; i < prob.length; i++) prob[i] += v[i];
  }
  for (let i = 0; i < prob.length; i++) prob[i] /= trees.length;
  let best = 0; for (let i = 1; i < prob.length; i++) if (prob[i] > prob[best]) best = i;
  const code = classes[best], info = labels[code];
  const all = {}; classes.forEach((c, i) => (all[c] = prob[i]));
  return { label: code, ko: info.ko, message: info.msg, confidence: prob[best], all };
}

const CAT_ICO = { B: "😻", F: "🍽️", I: "😿" };
const CAT_NAME = { B: "브러싱", F: "밥 기다림", I: "불안/외로움" };

function showCat(d) {
  window.lastCatAudio = d; // 영상 모듈과의 융합용
  document.getElementById("catResult").classList.add("show");
  document.getElementById("catIco").textContent = CAT_ICO[d.label] || "🐾";
  document.getElementById("catLabel").textContent = `${d.ko} (${Math.round(d.confidence * 100)}%)`;
  document.getElementById("catMsg").textContent = d.message;
  document.getElementById("catBars").innerHTML = Object.entries(d.all)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `<div class="row"><span>${CAT_ICO[k] || ""} ${CAT_NAME[k] || k}</span><span>${Math.round(v * 100)}%</span></div>
      <div class="bar"><div data-w="${(v * 100).toFixed(1)}"></div></div>`).join("");
  animateBars("catBars");
}

// 막대를 0에서 목표치로 애니메이션
function animateBars(id) {
  requestAnimationFrame(() => {
    document.querySelectorAll(`#${id} .bar > div`).forEach((el) => {
      el.style.width = el.dataset.w + "%";
    });
  });
}

function flatten(arr) {
  const len = arr.reduce((s, a) => s + a.length, 0);
  const out = new Float32Array(len); let off = 0;
  for (const a of arr) { out.set(a, off); off += a.length; }
  return out;
}

// ---- 사람 -> 고양이 (클라이언트, 재미용) ----
const HUMAN_TO_CAT = [
  { keys: ["밥", "사료", "먹", "간식", "배고"], ko: "밥 줄게", ico: "🍽️", meow: "므르르~ 야아아옹! (기대하는 소리)", sound: "food" },
  { keys: ["사랑", "예뻐", "귀여", "착해", "쓰다듬", "이뻐", "좋아"], ko: "애정 표현", ico: "😽", meow: "그르르릉~ (편안한 소리)", sound: "affection" },
  { keys: ["놀", "장난", "이리", "와봐"], ko: "놀자", ico: "🐈", meow: "먀악! 먀악! (들뜬 소리)", sound: "play" },
  { keys: ["미안", "괜찮", "안돼", "하지마", "진정"], ko: "진정", ico: "🙀", meow: "...야옹. (낮은 소리)", sound: "calm" },
];

const catAudio = new Audio();
catAudio.preload = "auto";
catAudio.volume = 0.9;
let lastSound = null;
const replayBtn = document.getElementById("replayBtn");

document.getElementById("sayBtn").onclick = () => {
  const text = document.getElementById("humanText").value.trim();
  if (!text) return;
  const hit = HUMAN_TO_CAT.find((m) => m.keys.some((k) => text.includes(k)))
    || { ko: "중립", ico: "🐱", meow: "야옹? (갸웃하는 소리)", sound: "neutral" };
  document.getElementById("humanResult").classList.add("show");
  document.getElementById("humanIco").textContent = hit.ico;
  document.getElementById("humanMeow").textContent = hit.meow;
  document.getElementById("humanKo").textContent = `의도: ${hit.ko}  🔊 재생 중`;
  document.getElementById("humanDisc").textContent =
    "※ 실제 고양이 울음(CatMeows 데이터)을 재생합니다. 사람→고양이 번역은 과학적 근거 없는 재미용 기능이에요.";
  lastSound = hit.sound;
  replayBtn.style.display = "block";
  playCat(hit.sound);
};

replayBtn.onclick = () => { if (lastSound) playCat(lastSound); };

function playCat(name) {
  catAudio.src = `./sounds/${name}.wav`;
  catAudio.currentTime = 0;
  catAudio.play().catch(() => {});
}
