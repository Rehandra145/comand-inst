/* ==========================================================================
   Robot Voice Command — Alpine.js controller
   - Mic robot (Jetson hardware) ATAU mic perangkat ini (HP/laptop via HTTPS)
   - Browser mic: getUserMedia → resample ke 16 kHz → WebSocket binary (float32)
   ========================================================================== */

const TARGET_RATE = 16000;
const CHUNK_SAMPLES = 1600; // 100 ms @ 16 kHz

document.addEventListener("alpine:init", () => {
  Alpine.data("zenVoiceSystem", () => ({
    // ---- koneksi & state ----
    connected: false,
    ws: null,
    state: "IDLE", // IDLE | LISTENING | PROCESSING | SPEAKING
    stateSince: Date.now(),
    starting: false,
    partialTranscript: "",
    robotResponse: "",
    lastNlu: null,
    audioLevel: 0,
    hint: "",
    showDrawer: false,
    panel: null, // null | "quick" | "type"
    textInput: "",

    // ---- mic ----
    isHttps: window.location.protocol === "https:",
    browserMicSupported: !!(window.isSecureContext && navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
    micSource: "browser",

    // internal browser-mic (tidak reaktif)
    _ctx: null,
    _stream: null,
    _node: null,
    _streaming: false,
    _pending: null,

    quickCommands: [
      { label: "Antar ke ICU", text: "tolong antar saya ke ruang ICU" },
      { label: "Bawa obat ke lab", text: "bawa obat ini ke laboratorium" },
      { label: "Maju 2 meter", text: "tolong gerak ke depan 2 meter" },
      { label: "Berhenti", text: "berhenti sekarang" },
      { label: "Kembali ke lobi", text: "kembali ke lobi utama" },
    ],

    telemetry: { device_model: "NVIDIA Jetson", active_accelerator: "-", cpu_load: 0, gpu_load: 0, ram_used_mb: 0, ram_total_mb: 0 },
    ros2: { mode: "mock" },

    _amp: 0,
    _phase: 0,

    // ---------------------------------------------------------------- init
    init() {
      const saved = localStorage.getItem("micSource");
      if (saved === "jetson" || saved === "browser") this.micSource = saved;
      if (!this.browserMicSupported) this.micSource = "jetson";
      if (!this.browserMicSupported && !this.isHttps) {
        this.hint = "Mic perangkat ini butuh HTTPS — buka lewat https:// (server --ssl).";
      }

      this.fetchInitialStatus();
      this.connectWebSocket();
      this.$nextTick(() => this.initWaveCanvas());
    },

    get statusLabel() {
      if (!this.connected) return "Terhubung ulang…";
      const s = { IDLE: "Siap", LISTENING: "Mendengarkan", PROCESSING: "Memproses", SPEAKING: "Robot berbicara" }[this.state] || this.state;
      return this.ros2.mode === "mock" ? `${s} · simulasi` : s;
    },

    get caption() {
      if (this.starting) return "Menyiapkan mikrofon…";
      switch (this.state) {
        case "LISTENING": return "Silakan bicara…";
        case "PROCESSING": return "Memahami perintah…";
        case "SPEAKING": return "Robot merespons…";
        default: return "Sentuh untuk bicara";
      }
    },

    setMicSource(src) {
      if (src === "browser" && !this.browserMicSupported) return;
      this.micSource = src;
      localStorage.setItem("micSource", src);
      this.hint = "";
    },

    togglePanel(name) {
      this.panel = this.panel === name ? null : name;
      if (this.panel === "type") this.$nextTick(() => this.$refs.textField && this.$refs.textField.focus());
    },

    setState(s) {
      if (s !== this.state) this.stateSince = Date.now();
      this.state = s;
    },

    // ---------------------------------------------------------------- network
    async fetchInitialStatus() {
      try {
        const res = await fetch("/api/status");
        if (!res.ok) return;
        const d = await res.json();
        this.setState(d.state || "IDLE");
        this.telemetry = { ...this.telemetry, ...d.telemetry };
        this.ros2 = { ...this.ros2, ...d.ros2 };
      } catch (e) {
        console.warn("status gagal", e);
      }
    },

    connectWebSocket() {
      const proto = this.isHttps ? "wss:" : "ws:";
      const ws = new WebSocket(`${proto}//${location.host}/ws`);
      ws.binaryType = "arraybuffer";
      this.ws = ws;

      ws.onopen = () => { this.connected = true; };
      ws.onmessage = (ev) => {
        if (typeof ev.data !== "string") return;
        try { this.onMessage(JSON.parse(ev.data)); } catch (e) { console.error(e); }
      };
      ws.onclose = () => {
        this.connected = false;
        this.starting = false;
        this._stopCapture();
        setTimeout(() => this.connectWebSocket(), 1500);
      };
    },

    // Log diagnostik: tampil di console browser DAN di terminal server
    dbg(...args) {
      const msg = args.map((a) => (typeof a === "string" ? a : JSON.stringify(a))).join(" ");
      console.log("[BrowserMic]", msg);
      this.send({ action: "client_log", msg });
    },

    send(obj) {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify(obj));
        return true;
      }
      return false;
    },

    onMessage(msg) {
      switch (msg.type) {
        case "handshake":
          this.setState(msg.state || "IDLE");
          this.ros2.mode = msg.mock ? "mock" : "socket";
          break;
        case "audio_level":
          if (!this._streaming) this.audioLevel = msg.level || 0; // mic robot
          break;
        case "telemetry":
          this.telemetry = { ...this.telemetry, ...msg.data };
          if (msg.state && msg.state !== this.state && !this.starting) this.setState(msg.state);
          break;
        case "state_changed":
          this.setState(msg.state);
          if (msg.state === "LISTENING") {
            this.partialTranscript = "";
            this.robotResponse = "";
            this._watchSilence();
          } else if (msg.state !== "LISTENING" && this._streaming) {
            this._stopCapture();
          }
          if (msg.state === "IDLE") this.audioLevel = 0;
          break;
        case "stt_partial":
          this.partialTranscript = msg.text;
          break;
        case "stt_final":
          if (this._streaming) this._stopCapture();
          if (msg.text) this.partialTranscript = msg.text;
          else if (msg.message) this.hint = msg.message;
          break;
        case "nlu_result":
          this.lastNlu = msg.data;
          this.robotResponse = msg.response || (msg.data && msg.data.robot_response) || "";
          this.hint = "";
          break;
        case "browser_mic_ready":
          this.dbg("browser_mic_ready", { success: msg.success, state: msg.state });
          this.starting = false;
          if (msg.success) {
            this._streaming = true;
            this._flushPending();
          } else {
            this._stopCapture();
            this.hint = "Sistem sedang sibuk, coba lagi sebentar.";
          }
          break;
        case "busy":
          this.hint = "Sistem sedang sibuk, coba lagi sebentar.";
          break;
        case "error":
          this.hint = msg.message || "Terjadi kesalahan.";
          break;
        case "ros2_mode_changed":
          this.ros2.mode = msg.mock ? "mock" : "socket";
          break;
      }
    },

    // ---------------------------------------------------------------- mic button
    toggleMic() {
      this.hint = "";
      if (!this.connected) {
        this.hint = "Belum terhubung ke robot. Menghubungkan ulang…";
        return;
      }
      if (this.starting) return;

      if (this.state === "LISTENING") {
        this.stopListening();
        return;
      }
      if (this.state === "PROCESSING" || this.state === "SPEAKING") {
        // Pemulihan otomatis bila state macet > 20 detik
        if (Date.now() - this.stateSince > 20000) {
          this.forceReset();
        } else {
          this.hint = "Tunggu sebentar, robot masih memproses.";
        }
        return;
      }
      this.startListening();
    },

    startListening() {
      this.partialTranscript = "";
      this.robotResponse = "";
      if (this.micSource === "browser") {
        this._startBrowserMic();
      } else {
        this.send({ action: "start_mic", duration: 10 });
      }
    },

    stopListening() {
      // Sesi bisa dimulai dari klien lain / sumber lain → hentikan keduanya
      if (this._node) {
        this._flushPending();
        this._stopCapture();
      }
      this.send({ action: "stop_browser_mic" });
      this.send({ action: "stop_mic" });
    },

    // Deteksi mic yang tidak mengirim sinyal sama sekali (mis. mic USB tidak terpasang)
    _watchSilence() {
      let peak = 0;
      const started = Date.now();
      const tick = () => {
        if (this.state !== "LISTENING") return;
        peak = Math.max(peak, this.audioLevel);
        if (Date.now() - started < 2500) return requestAnimationFrame(tick);
        if (peak < 0.003) {
          this.hint = this.micSource === "jetson"
            ? "Mic robot tidak menangkap sinyal. Pastikan mic USB terpasang di Jetson."
            : "Mikrofon perangkat ini tidak menangkap sinyal. Periksa mic / izin browser.";
        }
      };
      requestAnimationFrame(tick);
    },

    forceReset() {
      this._stopCapture();
      this.starting = false;
      this.send({ action: "reset" });
      this.hint = "";
    },

    // ---------------------------------------------------------------- browser mic
    // Harus dipanggil langsung dari gesture sentuhan (syarat iOS/Android).
    async _startBrowserMic() {
      this.starting = true;
      try {
        const AC = window.AudioContext || window.webkitAudioContext;
        // Buat AudioContext SEBELUM await agar masih dalam user gesture (iOS)
        this._ctx = new AC();
        if (this._ctx.state === "suspended") this._ctx.resume();

        this._stream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
        });

        const ctx = this._ctx;
        // Dialog izin mic sering membuat AudioContext ter-suspend (iOS/Android)
        if (ctx.state !== "running") {
          try { await ctx.resume(); } catch (_) {}
        }
        const track = this._stream.getAudioTracks()[0];
        this.dbg("mic ok", { label: track && track.label, ctxState: ctx.state, rate: ctx.sampleRate, ua: navigator.userAgent.slice(0, 80) });
        if (!this._ctx) return; // dibatalkan saat menunggu izin
        const src = ctx.createMediaStreamSource(this._stream);
        const node = ctx.createScriptProcessor(4096, 1, 1);
        const ratio = ctx.sampleRate / TARGET_RATE;
        let carry = 0; // posisi fraksional antar buffer
        this._pending = [];
        let acc = new Float32Array(CHUNK_SAMPLES);
        let accLen = 0;

        let frames = 0;
        node.onaudioprocess = (e) => {
          const input = e.inputBuffer.getChannelData(0);
          if (frames++ === 0) this.dbg("audio mengalir, ctxState=" + ctx.state);

          // level lokal untuk visual gelombang (responsif tanpa menunggu server)
          let sum = 0;
          for (let i = 0; i < input.length; i += 4) sum += input[i] * input[i];
          this.audioLevel = Math.min(1, Math.sqrt(sum / (input.length / 4)) * 18);

          // Resample (rata-rata per jendela) → 16 kHz
          let pos = carry;
          while (pos + ratio <= input.length) {
            const start = Math.max(0, Math.floor(pos));
            const end = Math.max(start + 1, Math.min(input.length, Math.floor(pos + ratio)));
            let s = 0;
            for (let i = start; i < end; i++) s += input[i];
            acc[accLen++] = s / (end - start);
            pos += ratio;
            if (accLen === CHUNK_SAMPLES) {
              this._emit(acc);
              acc = new Float32Array(CHUNK_SAMPLES);
              accLen = 0;
            }
          }
          carry = pos - input.length;
        };

        src.connect(node);
        // Sambungkan ke output lewat gain 0 agar tidak terdengar feedback
        const mute = ctx.createGain();
        mute.gain.value = 0;
        node.connect(mute);
        mute.connect(ctx.destination);
        this._node = node;

        // Minta server membuka sesi STT; audio di-buffer sampai siap
        if (!this.send({ action: "start_browser_mic" })) throw new Error("WebSocket terputus");
        this.dbg("start_browser_mic dikirim");
        // Peringatan bila audio tidak mengalir sama sekali
        setTimeout(() => {
          if (this._node && frames === 0) {
            this.dbg("TIDAK ADA audio frame, ctxState=" + (this._ctx && this._ctx.state));
            if (this._ctx) this._ctx.resume();
          }
        }, 1500);
      } catch (err) {
        console.error("[BrowserMic]", err);
        this.dbg("ERROR", (err && err.name) + ": " + (err && err.message));
        this.starting = false;
        this._stopCapture();
        const name = err && err.name;
        this.hint = name === "NotAllowedError"
          ? "Izin mikrofon ditolak. Izinkan mic di pengaturan browser."
          : name === "NotFoundError"
          ? "Mikrofon tidak ditemukan di perangkat ini."
          : "Gagal membuka mikrofon: " + (err.message || err);
      }
    },

    _emit(chunk) {
      if (this._streaming) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(chunk.buffer);
      } else if (this._pending && this._pending.length < 30) {
        this._pending.push(chunk); // tahan maks 3 detik saat menunggu server
      }
    },

    _flushPending() {
      if (!this._pending || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;
      for (const c of this._pending) this.ws.send(c.buffer);
      this._pending = [];
    },

    _stopCapture() {
      this._streaming = false;
      this._pending = null;
      if (this._node) { try { this._node.disconnect(); } catch (_) {} this._node.onaudioprocess = null; this._node = null; }
      if (this._stream) { this._stream.getTracks().forEach((t) => t.stop()); this._stream = null; }
      if (this._ctx) { this._ctx.close().catch(() => {}); this._ctx = null; }
      this.audioLevel = 0;
    },

    // ---------------------------------------------------------------- text
    submitText() {
      const q = this.textInput.trim();
      if (!q) return;
      this.sendTextCommand(q);
      this.textInput = "";
    },

    sendTextCommand(text) {
      this.partialTranscript = text;
      this.robotResponse = "";
      this.hint = "";
      if (!this.send({ action: "send_text", text })) {
        fetch("/api/nlu/process", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        }).then((r) => r.json()).then((d) => {
          this.lastNlu = d;
          this.robotResponse = d.robot_response || "";
        }).catch(() => (this.hint = "Gagal mengirim perintah."));
      }
    },

    toggleRos2Mode() {
      const mock = this.ros2.mode !== "mock";
      if (!this.send({ action: "set_mock", mock })) {
        fetch("/api/ros2/config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ mock }),
        }).then(() => (this.ros2.mode = mock ? "mock" : "socket"));
      }
    },

    // ---------------------------------------------------------------- waves
    initWaveCanvas() {
      const canvas = document.getElementById("waveCanvas");
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      let size = 0;

      const resize = () => {
        const dpr = window.devicePixelRatio || 1;
        size = canvas.clientWidth || 300;
        canvas.width = size * dpr;
        canvas.height = size * dpr;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      };
      resize();
      window.addEventListener("resize", resize);

      const layers = [
        { off: 10, f: 5, c: "rgba(218,119,86,0.50)", w: 2.2, p: 0 },
        { off: 22, f: 7, c: "rgba(232,168,146,0.35)", w: 1.8, p: 1.2 },
        { off: 36, f: 4, c: "rgba(212,163,89,0.25)", w: 1.4, p: 2.5 },
        { off: 52, f: 6, c: "rgba(204,120,92,0.14)", w: 1.0, p: 3.8 },
      ];

      const render = () => {
        const cx = size / 2, cy = size / 2;
        const base = size * 0.23;
        ctx.clearRect(0, 0, size, size);

        const listening = this.state === "LISTENING";
        const busy = this.state === "SPEAKING" || this.state === "PROCESSING";
        const target = listening ? Math.max(5, this.audioLevel * 42) : busy ? 7 : 0;
        this._amp += (target - this._amp) * 0.15;
        this._phase += listening ? 0.05 + this.audioLevel * 0.1 : 0.02;

        if (this._amp > 0.5) {
          for (const L of layers) {
            ctx.beginPath();
            const r0 = base + L.off;
            for (let i = 0; i <= 120; i++) {
              const t = (i / 120) * Math.PI * 2;
              const u = Math.sin(t * L.f + this._phase + L.p) * 0.7 + Math.cos(t * (L.f - 1) - this._phase * 0.8) * 0.3;
              const r = r0 + u * this._amp;
              const x = cx + Math.cos(t) * r, y = cy + Math.sin(t) * r;
              i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
            }
            ctx.closePath();
            ctx.strokeStyle = L.c;
            ctx.lineWidth = L.w;
            ctx.stroke();
          }
        } else {
          ctx.beginPath();
          ctx.arc(cx, cy, base + 8 + Math.sin(this._phase) * 3, 0, Math.PI * 2);
          ctx.strokeStyle = "rgba(204,120,92,0.12)";
          ctx.lineWidth = 1.2;
          ctx.stroke();
        }
        requestAnimationFrame(render);
      };
      render();
    },
  }));
});
