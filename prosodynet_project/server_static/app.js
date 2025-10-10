const form = document.getElementById("synth-form");
const synthBtn = document.getElementById("synth-btn");
const statusBox = document.getElementById("status");
const resultsSection = document.getElementById("results");
const neutralAudio = document.getElementById("neutral-audio");
const emotionalAudio = document.getElementById("emotional-audio");
const melLink = document.getElementById("mel-link");
const ckptLine = document.getElementById("ckpt-line");
const textInput = document.getElementById("text-input");
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
const useProsodynetInput = document.getElementById("use-prosodynet");
const prosodynetConfig = document.getElementById("prosodynet-config");
const vocoderSection = document.getElementById("vocoder-section");

const STORAGE_KEY = "prosodynet_ui_state";

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
    return checked ? checked.value : "coqui";
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
        }
    }
    if (state.orpheus_voice != null && orpheusVoiceSelect) {
        orpheusVoiceSelect.value = state.orpheus_voice;
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
    toggleBlock(coquiConfig, getTtsEngine() === "coqui");
    toggleBlock(orpheusConfig, getTtsEngine() === "orpheus");
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
        speaker_id: sanitize(speakerSelect.value) || "0001",
        emotion_id: Number.parseInt(emotionSelect.value, 10),
        tts_engine: getTtsEngine(),
        tts_model: sanitize(ttsModelInput.value) || "tts_models/multilingual/multi-dataset/xtts_v2",
        orpheus_voice: sanitize(orpheusVoiceSelect?.value) || "tara",
        use_prosodynet: useProsodynetInput?.checked ?? true,
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
    } catch (error) {
        console.error(error);
        setStatus(error.message, "error");
    } finally {
        synthBtn.disabled = false;
    }
});
