const form = document.getElementById("synth-form");
const synthBtn = document.getElementById("synth-btn");
const statusBox = document.getElementById("status");
const resultsSection = document.getElementById("results");
const neutralAudio = document.getElementById("neutral-audio");
const emotionalAudio = document.getElementById("emotional-audio");
const melLink = document.getElementById("mel-link");
const ckptLine = document.getElementById("ckpt-line");
const textInput = document.getElementById("text-input");
const exampleSelect = document.getElementById("example-select");
const applyExampleBtn = document.getElementById("apply-example-btn");
const speakerSelect = document.getElementById("speaker-select");
const emotionSelect = document.getElementById("emotion-select");
const ttsModelInput = document.getElementById("tts-model");
const hifiganFields = document.getElementById("hifigan-config");
const useRvcInput = document.getElementById("use-rvc");
const rvcFields = document.getElementById("rvc-config");
const rvcBaseInput = document.getElementById("rvc-base");
const rvcRefreshBtn = document.getElementById("rvc-refresh");
const rvcPthOptions = document.getElementById("rvc-pth-options");
const rvcConfigOptions = document.getElementById("rvc-config-options");
const rvcPthInput = document.getElementById("rvc-pth");
const rvcConfigInput = document.getElementById("rvc-config-path");
const vocoderRadios = Array.from(
    document.querySelectorAll('input[name="vocoder-mode"]')
);
const ttsEngineRadios = Array.from(
    document.querySelectorAll('input[name="tts-engine"]')
);
const coquiConfig = document.getElementById("coqui-config");
const orpheusConfig = document.getElementById("orpheus-config");
const orpheusVoiceSelect = document.getElementById("orpheus-voice");
const kokoroConfig = document.getElementById("kokoro-config");
const kokoroLangSelect = document.getElementById("kokoro-lang");
const kokoroVoiceSelect = document.getElementById("kokoro-voice");
const edgeConfig = document.getElementById("edge-config");
const edgeVoiceSelect = document.getElementById("edge-voice");
const gttsConfig = document.getElementById("gtts-config");
const gttsLangSelect = document.getElementById("gtts-lang");
const gttsTldSelect = document.getElementById("gtts-tld");
const pyttsx3Config = document.getElementById("pyttsx3-config");
const higgsConfig = document.getElementById("higgs-config");
const higgsTemperatureInput = document.getElementById("higgs-temperature");
const higgsTopPInput = document.getElementById("higgs-top-p");
const higgsMaxTokensInput = document.getElementById("higgs-max-tokens");
const fishConfig = document.getElementById("fish-config");
const fishTemperatureInput = document.getElementById("fish-temperature");
const fishTopPInput = document.getElementById("fish-top-p");
const fishMaxTokensInput = document.getElementById("fish-max-tokens");
const fishRepetitionPenaltyInput = document.getElementById("fish-repetition-penalty");
const useProsodynetInput = document.getElementById("use-prosodynet");
const prosodynetConfig = document.getElementById("prosodynet-config");
const vocoderSection = document.getElementById("vocoder-section");

const STORAGE_KEY = "prosodynet_ui_state_v2"; // Changed to reset old settings

function toggleBlock(element, show) {
    if (!element) return;
    element.classList.toggle("hidden", !show);
}

function setStatus(message, tone = "") {
    statusBox.textContent = message;
    statusBox.classList.remove("success", "error");
    if (tone) {
        statusBox.classList.add(tone);
    }
}

function getVocoderMode() {
    const checked = vocoderRadios.find((radio) => radio.checked);
    return checked ? checked.value : "griffinlim";
}

function getTtsEngine() {
    const checked = ttsEngineRadios.find((radio) => radio.checked);
    return checked ? checked.value : "edge";
}

function sanitize(value) {
    if (typeof value !== "string") {
        return null;
    }
    const trimmed = value.trim();
    return trimmed.length ? trimmed : null;
}

function cacheBust(path) {
    if (!path) return "";
    const separator = path.includes("?") ? "&" : "?";
    return `${path}${separator}t=${Date.now()}`;
}

function isAbsolutePath(value) {
    if (!value || typeof value !== "string") {
        return false;
    }
    const first = value[0];
    if (first === "/" || first === "\\") {
        return true;
    }
    if (value.length > 2 && value[1] === ":") {
        const third = value[2];
        return third === "/" || third === "\\";
    }
    return false;
}

function joinPaths(base, relative) {
    if (!base) return relative;
    const normalizedBase = base.replace(/[\\/]+$/, "");
    const normalizedRel = relative.replace(/^[\\/]+/, "");
    return `${normalizedBase}/${normalizedRel}`;
}

function loadState() {
    try {
        return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    } catch {
        return {};
    }
}

let state = loadState();

function saveState() {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (error) {
        console.warn("Failed to persist UI state:", error);
    }
}

function updateState(partial) {
    state = { ...state, ...partial };
    saveState();
}

function applyState() {
    if (state.text != null && textInput) {
        textInput.value = state.text;
    }
    if (state.tts_model != null && ttsModelInput) {
        ttsModelInput.value = state.tts_model;
    }
    if (state.speaker_id != null && speakerSelect) {
        speakerSelect.value = state.speaker_id;
    }
    if (state.emotion_id != null && emotionSelect) {
        emotionSelect.value = state.emotion_id;
    }
    if (state.vocoder_mode != null) {
        const radio = vocoderRadios.find((item) => item.value === state.vocoder_mode);
        if (radio) {
            radio.checked = true;
            toggleBlock(hifiganFields, radio.value === "hifigan");
        }
    }
    if (state.hifigan_module && document.getElementById("hifigan-module")) {
        document.getElementById("hifigan-module").value = state.hifigan_module;
    }
    if (state.hifigan_ckpt && document.getElementById("hifigan-ckpt")) {
        document.getElementById("hifigan-ckpt").value = state.hifigan_ckpt;
    }
    if (state.hifigan_config && document.getElementById("hifigan-config-path")) {
        document.getElementById("hifigan-config-path").value = state.hifigan_config;
    }
    if (state.use_rvc != null && useRvcInput) {
        useRvcInput.checked = state.use_rvc;
        toggleBlock(rvcFields, state.use_rvc);
    }
    if (state.rvc_base && rvcBaseInput) {
        rvcBaseInput.value = state.rvc_base;
    }
    if (state.rvc_cli && document.getElementById("rvc-cli")) {
        document.getElementById("rvc-cli").value = state.rvc_cli;
    }
    if (state.rvc_pth && rvcPthInput) {
        rvcPthInput.value = state.rvc_pth;
    }
    if (state.rvc_config && rvcConfigInput) {
        rvcConfigInput.value = state.rvc_config;
    }
    if (state.tts_engine != null) {
        const radio = ttsEngineRadios.find((item) => item.value === state.tts_engine);
        if (radio) {
            radio.checked = true;
            toggleBlock(coquiConfig, radio.value === "coqui");
            toggleBlock(orpheusConfig, radio.value === "orpheus");
            toggleBlock(kokoroConfig, radio.value === "kokoro");
            toggleBlock(edgeConfig, radio.value === "edge");
        }
    }
    if (state.orpheus_voice != null && orpheusVoiceSelect) {
        orpheusVoiceSelect.value = state.orpheus_voice;
    }
    if (state.kokoro_lang != null && kokoroLangSelect) {
        kokoroLangSelect.value = state.kokoro_lang;
    }
    if (state.kokoro_voice != null && kokoroVoiceSelect) {
        kokoroVoiceSelect.value = state.kokoro_voice;
    }
    if (state.edge_voice != null && edgeVoiceSelect) {
        edgeVoiceSelect.value = state.edge_voice;
    }
    if (state.gtts_lang != null && gttsLangSelect) {
        gttsLangSelect.value = state.gtts_lang;
    }
    if (state.gtts_tld != null && gttsTldSelect) {
        gttsTldSelect.value = state.gtts_tld;
    }
    if (state.use_prosodynet != null && useProsodynetInput) {
        useProsodynetInput.checked = state.use_prosodynet;
        toggleBlock(prosodynetConfig, state.use_prosodynet);
        toggleBlock(vocoderSection, state.use_prosodynet);
    }
}

function syncPanels() {
    toggleBlock(hifiganFields, getVocoderMode() === "hifigan");
    toggleBlock(rvcFields, !!useRvcInput?.checked);
    const engine = getTtsEngine();
    toggleBlock(coquiConfig, engine === "coqui");
    toggleBlock(orpheusConfig, engine === "orpheus");
    toggleBlock(kokoroConfig, engine === "kokoro");
    toggleBlock(edgeConfig, engine === "edge");
    toggleBlock(gttsConfig, engine === "gtts");
    toggleBlock(pyttsx3Config, engine === "pyttsx3");
    toggleBlock(higgsConfig, engine === "higgs");
    toggleBlock(fishConfig, engine === "fish");
    toggleBlock(prosodynetConfig, !!useProsodynetInput?.checked);
    toggleBlock(vocoderSection, !!useProsodynetInput?.checked);
}

applyState();
syncPanels();

// 시스템 정보 가져오기
async function fetchSystemInfo() {
    try {
        const response = await fetch("/system/info");
        if (!response.ok) {
            throw new Error(`Failed to fetch system info (HTTP ${response.status})`);
        }
        const data = await response.json();

        const cpuInfoEl = document.getElementById("cpu-info");
        if (cpuInfoEl) {
            const cpuText = `${data.cpu.cores}코어 (${data.cpu.model})`;
            cpuInfoEl.textContent = cpuText;
        }
    } catch (error) {
        console.warn("Could not load system info:", error);
        const cpuInfoEl = document.getElementById("cpu-info");
        if (cpuInfoEl) {
            cpuInfoEl.textContent = "정보 로드 실패";
        }
    }
}

// 페이지 로드 시 시스템 정보 가져오기
fetchSystemInfo();

function seedDefaultRvcValues() {
    const patch = {};
    const base = sanitize(rvcBaseInput?.value);
    if (base && state.rvc_base == null) {
        patch.rvc_base = base;
    }
    const cliValue = sanitize(document.getElementById("rvc-cli")?.value);
    if (cliValue && state.rvc_cli == null) {
        patch.rvc_cli = cliValue;
    }
    const pthValue = sanitize(rvcPthInput?.value);
    if (pthValue && state.rvc_pth == null) {
        patch.rvc_pth = pthValue;
    }
    const cfgValue = sanitize(rvcConfigInput?.value);
    if (cfgValue && state.rvc_config == null) {
        patch.rvc_config = cfgValue;
    }
    if (useRvcInput && state.use_rvc == null) {
        patch.use_rvc = useRvcInput.checked;
    }
    if (Object.keys(patch).length) {
        updateState(patch);
    }
}

seedDefaultRvcValues();

vocoderRadios.forEach((radio) => {
    radio.addEventListener("change", () => {
        const mode = getVocoderMode();
        syncPanels();
        updateState({ vocoder_mode: mode });
    });
    radio.addEventListener("input", syncPanels);
});

ttsEngineRadios.forEach((radio) => {
    radio.addEventListener("change", () => {
        const engine = getTtsEngine();

        syncPanels();
        updateState({ tts_engine: engine });
    });
    radio.addEventListener("input", syncPanels);
});

orpheusVoiceSelect?.addEventListener("change", () => {
    updateState({ orpheus_voice: orpheusVoiceSelect.value });
});

kokoroLangSelect?.addEventListener("change", () => {
    updateState({ kokoro_lang: kokoroLangSelect.value });
});

kokoroVoiceSelect?.addEventListener("change", () => {
    updateState({ kokoro_voice: kokoroVoiceSelect.value });
});

edgeVoiceSelect?.addEventListener("change", () => {
    updateState({ edge_voice: edgeVoiceSelect.value });
});

gttsLangSelect?.addEventListener("change", () => {
    updateState({ gtts_lang: gttsLangSelect.value });
});

gttsTldSelect?.addEventListener("change", () => {
    updateState({ gtts_tld: gttsTldSelect.value });
});

higgsTemperatureInput?.addEventListener("change", () => {
    updateState({ higgs_temperature: parseFloat(higgsTemperatureInput.value) });
});

higgsTopPInput?.addEventListener("change", () => {
    updateState({ higgs_top_p: parseFloat(higgsTopPInput.value) });
});

higgsMaxTokensInput?.addEventListener("change", () => {
    updateState({ higgs_max_tokens: parseInt(higgsMaxTokensInput.value) });
});

fishTemperatureInput?.addEventListener("change", () => {
    updateState({ fish_temperature: parseFloat(fishTemperatureInput.value) });
});

fishTopPInput?.addEventListener("change", () => {
    updateState({ fish_top_p: parseFloat(fishTopPInput.value) });
});

fishMaxTokensInput?.addEventListener("change", () => {
    updateState({ fish_max_tokens: parseInt(fishMaxTokensInput.value) });
});

fishRepetitionPenaltyInput?.addEventListener("change", () => {
    updateState({ fish_repetition_penalty: parseFloat(fishRepetitionPenaltyInput.value) });
});

useProsodynetInput?.addEventListener("change", () => {
    syncPanels();
    updateState({ use_prosodynet: useProsodynetInput.checked });
});

useProsodynetInput?.addEventListener("input", syncPanels);

useRvcInput.addEventListener("change", () => {
    syncPanels();
    updateState({ use_rvc: useRvcInput.checked });
    if (useRvcInput.checked && rvcPthOptions && !rvcPthOptions.children.length) {
        void refreshRvcFiles();
    }
});

useRvcInput.addEventListener("input", syncPanels);

async function refreshRvcFiles() {
    if (!rvcBaseInput) return;
    const base = sanitize(rvcBaseInput.value);
    if (!base) {
        setStatus("RVC 기본 경로를 입력해주세요.", "error");
        return;
    }

    try {
        setStatus("RVC 모델 목록을 불러오는 중…");
        if (rvcRefreshBtn) {
            rvcRefreshBtn.disabled = true;
        }
        const query = new URLSearchParams({ basePath: base }).toString();
        const response = await fetch(`/rvc/files?${query}`);
        if (!response.ok) {
            throw new Error(`RVC 경로 조회 실패 (HTTP ${response.status})`);
        }
        const data = await response.json();
        const basePath = data.base;

        populateDatalist(rvcPthOptions, basePath, data.pth || []);
        populateDatalist(rvcConfigOptions, basePath, data.config || []);

        if (rvcPthInput) {
            const current = sanitize(rvcPthInput.value);
            if (current && isAbsolutePath(current) && current.startsWith(basePath)) {
                const relative = current.slice(basePath.length).replace(/^[\\/]+/, "");
                rvcPthInput.value = relative;
                updateState({ rvc_pth: relative });
            }
        }
        if (rvcPthInput && !sanitize(rvcPthInput.value) && data.pth?.length) {
            rvcPthInput.value = data.pth[0];
            updateState({ rvc_pth: rvcPthInput.value });
        }
        if (rvcConfigInput) {
            const current = sanitize(rvcConfigInput.value);
            if (current && isAbsolutePath(current) && current.startsWith(basePath)) {
                const relative = current.slice(basePath.length).replace(/^[\\/]+/, "");
                rvcConfigInput.value = relative;
                updateState({ rvc_config: relative });
            }
        }
        if (rvcConfigInput && !sanitize(rvcConfigInput.value) && data.config?.length) {
            rvcConfigInput.value = data.config[0];
            updateState({ rvc_config: rvcConfigInput.value });
        }

        setStatus(
            `RVC 목록 업데이트 완료 (모델 ${data.pth?.length ?? 0}개, 설정 ${data.config?.length ?? 0}개)`,
            "success"
        );
        updateState({ rvc_base: basePath });
    } catch (error) {
        console.error(error);
        setStatus(error.message, "error");
    } finally {
        if (rvcRefreshBtn) {
            rvcRefreshBtn.disabled = false;
        }
    }
}

function populateDatalist(datalist, basePath, items) {
    if (!datalist) return;
    datalist.innerHTML = "";
    datalist.dataset.base = basePath;
    items.forEach((item) => {
        const option = document.createElement("option");
        option.value = item;
        option.label = item;
        datalist.appendChild(option);
    });
}

rvcRefreshBtn?.addEventListener("click", () => {
    void refreshRvcFiles();
});

textInput?.addEventListener("input", () => {
    updateState({ text: textInput.value });
});

// 예문 적용 기능
applyExampleBtn?.addEventListener("click", () => {
    const selectedExample = exampleSelect?.value;
    if (selectedExample && textInput) {
        textInput.value = selectedExample;
        updateState({ text: selectedExample });
        setStatus("예문이 적용되었습니다.", "success");

        // 선택 초기화
        if (exampleSelect) {
            exampleSelect.value = "";
        }
    }
});

// 예문 선택 시 직접 적용 (엔터키나 더블클릭)
exampleSelect?.addEventListener("dblclick", () => {
    applyExampleBtn?.click();
});

speakerSelect?.addEventListener("change", () => {
    updateState({ speaker_id: speakerSelect.value });
});

emotionSelect?.addEventListener("change", () => {
    updateState({ emotion_id: emotionSelect.value });
});

ttsModelInput?.addEventListener("input", () => {
    updateState({ tts_model: ttsModelInput.value });
});

document.getElementById("hifigan-module")?.addEventListener("input", (event) => {
    updateState({ hifigan_module: event.target.value });
});

document.getElementById("hifigan-ckpt")?.addEventListener("input", (event) => {
    updateState({ hifigan_ckpt: event.target.value });
});

document.getElementById("hifigan-config-path")?.addEventListener("input", (event) => {
    updateState({ hifigan_config: event.target.value });
});

rvcBaseInput?.addEventListener("input", () => {
    updateState({ rvc_base: rvcBaseInput.value });
});

document.getElementById("rvc-cli")?.addEventListener("input", (event) => {
    updateState({ rvc_cli: event.target.value });
});

rvcPthInput?.addEventListener("input", () => {
    updateState({ rvc_pth: rvcPthInput.value });
});

rvcConfigInput?.addEventListener("input", () => {
    updateState({ rvc_config: rvcConfigInput.value });
});

if (useRvcInput.checked && rvcPthOptions && !rvcPthOptions.children.length) {
    void refreshRvcFiles();
}

// 전체 다운로드 기능
const downloadAllBtn = document.getElementById("download-all-btn");
const browseFilesBtn = document.getElementById("browse-files-btn");
const fileBrowser = document.getElementById("file-browser");
const fileList = document.getElementById("file-list");

// ProsodyNet 파일 섹션
const refreshProsodynetBtn = document.getElementById("refresh-prosodynet-btn");
const prosodynetFilesSection = document.getElementById("prosodynet-files");
const prosodynetList = document.getElementById("prosodynet-list");

let currentResults = {
    neutral_wav: null,
    mel: null,
    emotional_wav: null
};

downloadAllBtn?.addEventListener("click", () => {
    const files = [];
    if (currentResults.neutral_wav) files.push({ url: currentResults.neutral_wav, name: "neutral.wav" });
    if (currentResults.mel) files.push({ url: currentResults.mel, name: "emotional.npy" });
    if (currentResults.emotional_wav) files.push({ url: currentResults.emotional_wav, name: "emotional.wav" });

    if (files.length === 0) {
        setStatus("다운로드할 파일이 없습니다.", "error");
        return;
    }

    files.forEach(file => {
        const a = document.createElement("a");
        a.href = file.url;
        a.download = file.name;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    });

    setStatus(`${files.length}개 파일 다운로드 시작`, "success");
});

browseFilesBtn?.addEventListener("click", async () => {
    try {
        setStatus("파일 목록 불러오는 중...");
        const response = await fetch("/static/list");

        if (!response.ok) {
            throw new Error(`파일 목록 조회 실패 (HTTP ${response.status})`);
        }

        const data = await response.json();
        displayFileList(data.files || []);
        fileBrowser.classList.remove("hidden");
        setStatus("파일 목록 로드 완료", "success");
    } catch (error) {
        console.error(error);
        setStatus(error.message, "error");
    }
});

function displayFileList(files) {
    if (!fileList) return;

    fileList.innerHTML = "";

    if (files.length === 0) {
        fileList.innerHTML = "<p>파일이 없습니다.</p>";
        return;
    }

    const ul = document.createElement("ul");
    files.forEach(file => {
        const li = document.createElement("li");

        const link = document.createElement("a");
        link.href = `/static/${file}`;
        link.textContent = file;
        link.download = file;

        const playBtn = document.createElement("button");
        playBtn.textContent = "▶️";
        playBtn.className = "play-btn";
        playBtn.onclick = (e) => {
            e.preventDefault();
            if (file.endsWith(".wav")) {
                neutralAudio.src = cacheBust(`/static/${file}`);
                neutralAudio.load();
                neutralAudio.play();
            }
        };

        li.appendChild(playBtn);
        li.appendChild(link);
        ul.appendChild(li);
    });

    fileList.appendChild(ul);
}

// ProsodyNet 파일 목록 표시
function displayProsodynetFiles(files) {
    if (!prosodynetList) return;

    prosodynetList.innerHTML = "";

    // neutral, emel, emotional 파일들을 그룹화
    const groups = new Map();

    files.forEach(file => {
        if (file.startsWith("neutral_")) {
            const id = file.replace("neutral_", "").replace(".wav", "");
            if (!groups.has(id)) groups.set(id, {});
            groups.get(id).neutral = file;
        } else if (file.startsWith("emel_")) {
            const id = file.replace("emel_", "").replace(".npy", "");
            if (!groups.has(id)) groups.set(id, {});
            groups.get(id).mel = file;
        } else if (file.startsWith("emotional_")) {
            const id = file.replace("emotional_", "").replace(".wav", "");
            if (!groups.has(id)) groups.set(id, {});
            groups.get(id).emotional = file;
        }
    });

    // emotional 파일이 있는 그룹만 표시 (ProsodyNet 사용한 것들)
    const prosodynetGroups = Array.from(groups.entries())
        .filter(([id, group]) => group.emotional)
        .sort((a, b) => b[0].localeCompare(a[0])); // 최신순 정렬

    if (prosodynetGroups.length === 0) {
        prosodynetList.innerHTML = "<p style='color: #94a3b8;'>ProsodyNet으로 생성된 파일이 없습니다.</p>";
        return;
    }

    const container = document.createElement("div");

    prosodynetGroups.forEach(([id, group]) => {
        const groupDiv = document.createElement("div");
        groupDiv.style.cssText = "margin-bottom: 20px; padding: 16px; background: rgba(30, 41, 59, 0.5); border-radius: 12px; border: 1px solid rgba(148, 163, 184, 0.2);";

        const title = document.createElement("div");
        title.style.cssText = "font-weight: 600; color: #e2e8f0; margin-bottom: 12px; font-size: 0.9rem;";
        title.textContent = `파일 ID: ${id.substring(0, 8)}...`;
        groupDiv.appendChild(title);

        const fileList = document.createElement("div");
        fileList.style.cssText = "display: flex; flex-direction: column; gap: 8px;";

        // Neutral 음성
        if (group.neutral) {
            const neutralDiv = document.createElement("div");
            neutralDiv.style.cssText = "display: flex; flex-direction: column; gap: 8px; margin-bottom: 8px;";

            const neutralHeader = document.createElement("div");
            neutralHeader.style.cssText = "display: flex; align-items: center; gap: 12px;";

            const neutralLabel = document.createElement("span");
            neutralLabel.textContent = "중립 음성:";
            neutralLabel.style.cssText = "color: #94a3b8; min-width: 100px; font-weight: 600;";

            const downloadLink = document.createElement("a");
            downloadLink.href = `/static/${group.neutral}`;
            downloadLink.download = group.neutral;
            downloadLink.textContent = "📥 다운로드";
            downloadLink.style.cssText = "color: #38bdf8; text-decoration: none;";

            neutralHeader.appendChild(neutralLabel);
            neutralHeader.appendChild(downloadLink);

            // 인라인 오디오 플레이어
            const audioPlayer = document.createElement("audio");
            audioPlayer.controls = true;
            audioPlayer.src = cacheBust(`/static/${group.neutral}`);
            audioPlayer.style.cssText = "width: 100%; max-width: 500px; border-radius: 8px;";

            neutralDiv.appendChild(neutralHeader);
            neutralDiv.appendChild(audioPlayer);
            fileList.appendChild(neutralDiv);
        }

        // Emotional 음성
        if (group.emotional) {
            const emotionalDiv = document.createElement("div");
            emotionalDiv.style.cssText = "display: flex; flex-direction: column; gap: 8px; margin-bottom: 8px; padding: 12px; background: rgba(34, 197, 94, 0.05); border-radius: 8px; border: 1px solid rgba(34, 197, 94, 0.2);";

            const emotionalHeader = document.createElement("div");
            emotionalHeader.style.cssText = "display: flex; align-items: center; gap: 12px;";

            const emotionalLabel = document.createElement("span");
            emotionalLabel.textContent = "🎭 감정 음성:";
            emotionalLabel.style.cssText = "color: #4ade80; min-width: 100px; font-weight: 700; font-size: 1.05rem;";

            const downloadLink = document.createElement("a");
            downloadLink.href = `/static/${group.emotional}`;
            downloadLink.download = group.emotional;
            downloadLink.textContent = "📥 다운로드";
            downloadLink.style.cssText = "color: #4ade80; text-decoration: none; font-weight: 600;";

            emotionalHeader.appendChild(emotionalLabel);
            emotionalHeader.appendChild(downloadLink);

            // 인라인 오디오 플레이어
            const audioPlayer = document.createElement("audio");
            audioPlayer.controls = true;
            audioPlayer.src = cacheBust(`/static/${group.emotional}`);
            audioPlayer.style.cssText = "width: 100%; max-width: 500px; border-radius: 8px;";

            emotionalDiv.appendChild(emotionalHeader);
            emotionalDiv.appendChild(audioPlayer);
            fileList.appendChild(emotionalDiv);
        }

        // Mel spectrogram
        if (group.mel) {
            const melDiv = document.createElement("div");
            melDiv.style.cssText = "display: flex; align-items: center; gap: 12px;";

            const melLabel = document.createElement("span");
            melLabel.textContent = "멜 스펙트로그램:";
            melLabel.style.cssText = "color: #94a3b8; min-width: 100px;";

            const downloadLink = document.createElement("a");
            downloadLink.href = `/static/${group.mel}`;
            downloadLink.download = group.mel;
            downloadLink.textContent = "📥 다운로드 (.npy)";
            downloadLink.style.cssText = "color: #38bdf8; text-decoration: none;";

            melDiv.appendChild(melLabel);
            melDiv.appendChild(downloadLink);
            fileList.appendChild(melDiv);
        }

        groupDiv.appendChild(fileList);
        container.appendChild(groupDiv);
    });

    prosodynetList.appendChild(container);
}

// ProsodyNet 파일 새로고침 버튼
refreshProsodynetBtn?.addEventListener("click", async () => {
    try {
        setStatus("ProsodyNet 파일 목록 불러오는 중...");
        const response = await fetch("/static/list");

        if (!response.ok) {
            throw new Error(`파일 목록 조회 실패 (HTTP ${response.status})`);
        }

        const data = await response.json();
        displayProsodynetFiles(data.files || []);
        prosodynetFilesSection.classList.remove("hidden");
        setStatus("ProsodyNet 파일 목록 로드 완료", "success");
    } catch (error) {
        console.error(error);
        setStatus(error.message, "error");
    }
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const text = sanitize(textInput.value);
    if (!text) {
        setStatus("문장을 입력해주세요.", "error");
        return;
    }

    setStatus("합성 중입니다. 잠시만 기다려주세요…");
    synthBtn.disabled = true;
    resultsSection.classList.add("hidden");

    const payload = {
        text,
        emotion_id: Number.parseInt(emotionSelect.value, 10),
        tts_engine: getTtsEngine(),
        tts_model: sanitize(ttsModelInput.value) || "tts_models/multilingual/multi-dataset/xtts_v2",
        speaker: sanitize(speakerSelect.value) || null,
        language: null,
        orpheus_voice: sanitize(orpheusVoiceSelect?.value) || "tara",
        kokoro_lang: sanitize(kokoroLangSelect?.value) || "j",
        kokoro_voice: sanitize(kokoroVoiceSelect?.value) || "jf_alpha",
        edge_voice: sanitize(edgeVoiceSelect?.value) || "ko-KR-SunHiNeural",
        edge_rate: "+0%",
        edge_pitch: "+0Hz",
        gtts_lang: sanitize(gttsLangSelect?.value) || "ko",
        gtts_tld: sanitize(gttsTldSelect?.value) || "com",
        higgs_temperature: parseFloat(higgsTemperatureInput?.value) || 0.3,
        higgs_top_p: parseFloat(higgsTopPInput?.value) || 0.95,
        higgs_max_tokens: parseInt(higgsMaxTokensInput?.value) || 1024,
        fish_temperature: parseFloat(fishTemperatureInput?.value) || 0.7,
        fish_top_p: parseFloat(fishTopPInput?.value) || 0.7,
        fish_max_tokens: parseInt(fishMaxTokensInput?.value) || 1024,
        fish_repetition_penalty: parseFloat(fishRepetitionPenaltyInput?.value) || 1.2,
        use_prosodynet: useProsodynetInput?.checked ?? false,
        use_rvc: useRvcInput.checked,
        vocoder: {
            mode: getVocoderMode()
        }
    };

    if (payload.vocoder.mode === "hifigan") {
        payload.vocoder.generator_module = sanitize(
            document.getElementById("hifigan-module").value
        );
        payload.vocoder.generator_ckpt = sanitize(
            document.getElementById("hifigan-ckpt").value
        );
        payload.vocoder.config = sanitize(
            document.getElementById("hifigan-config-path").value
        );
    }

    if (!payload.use_rvc) {
        payload.rvc = null;
    } else {
        const base = sanitize(rvcBaseInput?.value || "");
        const resolve = (value) => {
            if (!value) return null;
            if (isAbsolutePath(value)) {
                return value;
            }
            if (!base) {
                return value;
            }
            return joinPaths(base, value);
        };
        payload.rvc = {
            cli: sanitize(document.getElementById("rvc-cli").value),
            pth: resolve(sanitize(rvcPthInput.value)),
            config: resolve(sanitize(rvcConfigInput.value))
        };
    }

    try {
        const response = await fetch("/synthesize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`합성 실패 (HTTP ${response.status})`);
        }

        const data = await response.json();

        if (!data.neutral_wav) {
            throw new Error("서버가 유효한 결과를 반환하지 않았습니다.");
        }

        // 결과 저장
        currentResults = {
            neutral_wav: data.neutral_wav,
            mel: data.mel,
            emotional_wav: data.emotional_wav
        };

        neutralAudio.src = cacheBust(data.neutral_wav);
        neutralAudio.load();
        melLink.href = data.mel ? cacheBust(data.mel) : "#";
        melLink.classList.toggle("hidden", !data.mel);
        emotionalAudio.src = data.emotional_wav ? cacheBust(data.emotional_wav) : "";
        if (data.emotional_wav) {
            emotionalAudio.load();
        }
        emotionalAudio.parentElement.classList.toggle("hidden", !data.emotional_wav);

        if (data.ckpt_used) {
            ckptLine.textContent = `사용한 체크포인트: ${data.ckpt_used}`;
            ckptLine.classList.remove("hidden");
        } else {
            ckptLine.textContent = "";
            ckptLine.classList.add("hidden");
        }

        resultsSection.classList.remove("hidden");
        setStatus("합성 완료! 재생 버튼을 눌러 확인하세요.", "success");

        // ProsodyNet 사용된 경우 자동으로 ProsodyNet 섹션 표시
        if (data.prosodynet_enabled) {
            setTimeout(async () => {
                try {
                    const response = await fetch("/static/list");
                    if (response.ok) {
                        const listData = await response.json();
                        displayProsodynetFiles(listData.files || []);
                        prosodynetFilesSection.classList.remove("hidden");
                    }
                } catch (err) {
                    console.error("ProsodyNet 파일 목록 자동 갱신 실패:", err);
                }
            }, 500);
        }
    } catch (error) {
        console.error(error);
        setStatus(error.message, "error");
    } finally {
        synthBtn.disabled = false;
    }
});
