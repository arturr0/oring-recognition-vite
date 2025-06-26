importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');

console.log("[Worker] Starting...");

const supportsWebGPU = !!navigator.gpu;

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
ort.env.wasm.simd = true;
ort.env.wasm.numThreads = 4;
// Optional: backendHint removed to avoid conflict

let session = null;
let processing = false;

self.onmessage = async (e) => {
    const { type, modelUrl, tensorData, dims } = e.data;

    if (type === 'loadModel') {
        try {
            console.log('[Worker] Loading model...');
            const startLoad = performance.now();

            const sessionOptions = {
                executionProviders: supportsWebGPU ? ['webgpu', 'wasm'] : ['wasm'],
                graphOptimizationLevel: 'all',
                enableCpuMemArena: true,
                enableMemPattern: true,
                executionMode: 'sequential',
                enableProfiling: false
            };

            try {
                session = await ort.InferenceSession.create(modelUrl, sessionOptions);
            } catch (firstError) {
                console.warn('[Worker] First load attempt failed, trying fallback...');
                sessionOptions.externalData = false;
                session = await ort.InferenceSession.create(modelUrl, sessionOptions);
            }

            console.log(`[Worker] Model loaded in ${(performance.now() - startLoad).toFixed(1)} ms`);
            self.postMessage({ type: 'loaded' });
        } catch (err) {
            self.postMessage({ type: 'error', message: `Model load failed: ${err.message}` });
        }
    }

    if (type === 'infer') {
        if (!session || processing) return;

        processing = true;
        try {
            const startInfer = performance.now();
            const tensor = new ort.Tensor('float32', new Float32Array(tensorData), dims);
            const feeds = { [session.inputNames[0]]: tensor };
            const results = await session.run(feeds);

            const output = results[session.outputNames[0]];
            self.postMessage({
                type: 'inference',
                data: output.data.buffer,
                dims: output.dims,
                inferTime: (performance.now() - startInfer).toFixed(1),
            }, [output.data.buffer]);
        } catch (err) {
            self.postMessage({ type: 'error', message: `Inference failed: ${err.message}` });
        } finally {
            processing = false;
        }
    }
};
