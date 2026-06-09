// 완전 클라이언트용: 녹음 -> dsp.js 특징 -> model.json RandomForest 추론 (서버 불필요)
import { extractFeatures } from "./lib/dsp.js";

let recording = false, audioCtx, stream, recNode, source, chunks = [], sampleRate = 16000;
let MODEL = null, recTimer = null, recStart = 0, maxRecTimer = null;

const MAX_REC_SEC = 15; // 자동 중지 (메모리 보호)
const MODEL_VER = "3";  // model.json 갱신 시 올려서 캐시 무효화

const recBtn = document.getElementById("recBtn");
const catStatus = document.getElementById("catStatus");

// 모델 로드 — 완료 전엔 녹음 버튼 비활성
recBtn.disabled = true;
recBtn.textContent = "모델 로딩 중...";
fetch(`./model.json?v=${MODEL_VER}`).then((r) => r.json()).then((m) => {
  MODEL = m;
  recBtn.disabled = false;
  recBtn.textContent = "녹음 시작";
}).catch(() => { catStatus.textContent = "모델 로드 실패 — 인터넷 연결을 확인하고 새로고침해 주세요."; });

recBtn.onclick = async () => { if (!recording) await startRec(); else await stopRec(); };

// AudioWorklet 녹음 노드 (구형 브라우저는 ScriptProcessor 폴백)
async function makeRecorder(ctx) {
  if (ctx.audioWorklet) {
    const code = `class Rec extends AudioWorkletProcessor {
      process(inputs) {
        const ch = inputs[0][0];
        if (ch) this.port.postMessage(ch.slice(0));
        return true;
      }
    } registerProcessor("rec", Rec);`;
    const url = URL.createObjectURL(new Blob([code], { type: "application/javascript" }));
    try {
      await ctx.audioWorklet.addModule(url);
      const node = new AudioWorkletNode(ctx, "rec");
      node.port.onmessage = (e) => { if (recording) chunks.push(e.data); };
      return node;
    } finally { URL.revokeObjectURL(url); }
  }
  const sp = ctx.createScriptProcessor(4096, 1, 1);
  sp.onaudioprocess = (e) => { if (recording) chunks.push(new Float32Array(e.inputBuffer.getChannelData(0))); };
  return sp;
}

async function startRec() {
  try {
    // 음성용 자동처리 끄기 (고양이 소리/음량을 망가뜨리지 않도록)
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
  } catch { catStatus.textContent = "마이크 권한이 필요합니다."; return; }
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  await audioCtx.resume(); // iOS Safari: suspended 상태 해제
  sampleRate = audioCtx.sampleRate;
  source = audioCtx.createMediaStreamSource(stream);
  chunks = [];
  recording = true;
  recNode = await makeRecorder(audioCtx);
  source.connect(recNode); recNode.connect(audioCtx.destination);
  recBtn.innerHTML = "■ 녹음 중지"; recBtn.classList.add("rec");
  recStart = Date.now();
  const tick = () => {
    const s = ((Date.now() - recStart) / 1000).toFixed(1);
    catStatus.innerHTML = `<span class="rec-dot"></span>녹음 중 ${s}s · 고양이 소리를 들려주세요 (최대 ${MAX_REC_SEC}초)`;
  };
  tick(); recTimer = setInterval(tick, 100);
  maxRecTimer = setTimeout(() => { if (recording) stopRec(); }, MAX_REC_SEC * 1000);
}

async function stopRec() {
  recording = false;
  clearInterval(recTimer); clearTimeout(maxRecTimer);
  recBtn.innerHTML = "녹음 시작"; recBtn.classList.remove("rec");
  recNode.disconnect(); source.disconnect();
  stream.getTracks().forEach((t) => t.stop());
  await audioCtx.close();

  catStatus.innerHTML = `<span class="spin"></span>분석 중...`;
  // 스피너가 실제로 그려지도록 한 프레임 양보 후 무거운 계산 시작
  await new Promise((r) => setTimeout(r, 50));

  const pcm = flatten(chunks);
  if (pcm.length < sampleRate * 0.2) { catStatus.textContent = "녹음이 너무 짧아요. 다시 시도해 주세요."; return; }
  // 소리 활동 구간(첫~마지막 울음)만 잘라 분석 — 무음·잡음 구간 제거
  const seg = pickActiveSpan(pcm, sampleRate);
  const feats = extractFeatures(seg, sampleRate);
  showCat(predict(feats));
  catStatus.textContent = "";
}

// RMS 기준 활동 구간(첫 소리~마지막 소리)을 보존해 잘라냄.
// 학습 클립이 울음 전체(연속 울음 포함)라서 한 구간만 자르면 오히려 어긋남 — 내장 샘플로 검증된 방식.
function pickActiveSpan(pcm, sr) {
  const frame = Math.round(sr * 0.05); // 50ms
  const n = Math.floor(pcm.length / frame);
  if (n < 4) return pcm;
  const rms = new Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = i * frame; j < (i + 1) * frame; j++) s += pcm[j] * pcm[j];
    rms[i] = Math.sqrt(s / frame);
  }
  let pi = 0;
  for (let i = 1; i < n; i++) if (rms[i] > rms[pi]) pi = i;
  if (rms[pi] < 1e-4) return pcm;
  const thr = rms[pi] * 0.03;
  let first = -1, last = -1;
  for (let i = 0; i < n; i++) if (rms[i] > thr) { if (first < 0) first = i; last = i; }
  if (first < 0) return pcm;
  const pad = Math.round(sr * 0.1);
  let a = Math.max(0, first * frame - pad), b = Math.min(pcm.length, (last + 1) * frame + pad);
  const maxLen = sr * 6; // 너무 길면 피크 중심으로 6초만
  if (b - a > maxLen) {
    const c = Math.round((pi + 0.5) * frame);
    a = Math.max(a, Math.min(c - maxLen / 2, b - maxLen)); b = a + maxLen;
  }
  return pcm.slice(a, b);
}

// ---- 내장 샘플로 모델 점검 (도메인 내 깨끗한 소리엔 잘 맞는지 확인) ----
const SAMPLE_TESTS = [
  { file: "food", expect: "F", name: "밥 기다림 소리" },
  { file: "affection", expect: "B", name: "브러싱 소리" },
  { file: "calm", expect: "I", name: "불안(격리) 소리" },
];
document.getElementById("sampleBtn").onclick = async () => {
  if (!MODEL) { catStatus.textContent = "모델 로딩 중..."; return; }
  catStatus.innerHTML = `<span class="spin"></span>내장 샘플 점검 중...`;
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const lines = [];
  for (const t of SAMPLE_TESTS) {
    try {
      const buf = await fetch(`./sounds/${t.file}.wav`).then((r) => r.arrayBuffer());
      const audio = await ctx.decodeAudioData(buf);
      const pcm = audio.getChannelData(0);
      const r = predict(extractFeatures(pcm, audio.sampleRate));
      const ok = r.label === t.expect;
      lines.push(`${ok ? "✅" : "❌"} ${t.name} → ${r.ko} (${Math.round(r.confidence * 100)}%)`);
    } catch (e) { lines.push(`⚠️ ${t.name}: ${e.message}`); }
  }
  ctx.close();
  document.getElementById("catResult").classList.add("show");
  document.getElementById("catIco").textContent = "🔎";
  document.getElementById("catLabel").textContent = "모델 점검 결과";
  document.getElementById("catMsg").innerHTML = lines.join("<br>") +
    "<br><span style='color:var(--muted);font-size:.8rem'>샘플이 잘 맞으면 모델은 정상 — 실제 녹음이 안 맞는 건 마이크/환경 차이 때문이에요.</span>";
  document.getElementById("catBars").innerHTML = "";
  catStatus.textContent = "";
};

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

  // 확신 게이팅: 1·2위 확률 차가 작으면 "불확실"로 정직하게 표시
  const sorted = Object.values(d.all).sort((a, b) => b - a);
  const margin = sorted[0] - (sorted[1] || 0);
  const uncertain = d.confidence < 0.5 || margin < 0.15;

  document.getElementById("catIco").textContent = uncertain ? "🤔" : (CAT_ICO[d.label] || "🐾");
  document.getElementById("catLabel").textContent = uncertain
    ? `잘 모르겠어요 (${d.ko} 추정 ${Math.round(d.confidence * 100)}%)`
    : `${d.ko} (${Math.round(d.confidence * 100)}%)`;
  document.getElementById("catMsg").textContent = uncertain
    ? "소리가 또렷하지 않거나 학습한 상황과 달라요. 조용한 곳에서 고양이에게 가까이 대고 1~2초 녹음해 보세요."
    : d.message;
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
