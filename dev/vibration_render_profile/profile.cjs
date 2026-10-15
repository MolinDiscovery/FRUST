// Run with NODE_PATH pointing to a Playwright installation; see the profile report.
const {chromium} = require('playwright');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {execFileSync} = require('node:child_process');

(async () => {
    const [notebook, seconds = '360', ...flags] = process.argv.slice(2);
    if (!notebook) throw Error('Usage: profile.cjs notebook.ipynb|grid.html [seconds] [--fixed]');
    let html = fs.readFileSync(notebook, 'utf8');
    if (path.extname(notebook) !== '.html') {
        const cells = JSON.parse(html).cells;
        const output = cells.flatMap(cell => cell.outputs || []).find(output =>
            output.data?.['application/3dmoljs_load.v0']);
        if (!output) throw Error('No saved 3Dmol output found');
        const raw = output.data['text/html'];
        html = Array.isArray(raw) ? raw.join('') : raw;
    }
    const viewerName = html.match(/var (viewer_\d+) = null/)[1];
    if (flags.includes('--fixed')) {
        const helper = execFileSync('conda', ['run', '-n', 'UMA', 'python', '-c',
            'from tooltoad.scene3d import Py3DmolGridRenderer; print(Py3DmolGridRenderer._VIBRATION_RENDER_CLEANUP)'],
            {encoding: 'utf8'}).trim();
        html = html.replace(/(viewergrid_\d+\[\d+\]\[\d+\])\.animate\(/g,
            (match, viewer) => `(${helper})(${viewer});\n${match}`);
    }
    const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'frust-vibration-'));
    fs.writeFileSync(path.join(temp, 'grid.html'), html);
    const scriptURL = html.match(/loadScriptAsync\('([^']+)'\)/)[1];
    const response = await fetch(scriptURL);
    if (!response.ok) throw Error(`Cannot fetch ${scriptURL}: ${response.status}`);
    const library = await response.text();
    const browser = await chromium.launch({
        executablePath: process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        headless: true, args: ['--disable-background-timer-throttling'],
    });
    try {
        const page = await browser.newPage({viewport: {width: 1200, height: 950}});
        page.on('pageerror', error => console.error(error));
        await page.addInitScript(() => {
            window.profile = {gl: {}, pending: new Map(), frames: 0};
            const set = window.setTimeout, clear = window.clearTimeout;
            window.setTimeout = function(fn, ms, ...args) {
                if (typeof fn !== 'function') return set(fn, ms, ...args);
                const id = set(() => { profile.pending.delete(id); fn(...args); }, ms);
                profile.pending.set(id, fn);
                return id;
            };
            window.clearTimeout = function(id) { profile.pending.delete(id); return clear(id); };
            for (const prototype of [WebGLRenderingContext.prototype, WebGL2RenderingContext.prototype]) {
                for (const name of ['createBuffer', 'deleteBuffer', 'createProgram', 'deleteProgram',
                    'createTexture', 'deleteTexture', 'drawElements']) {
                    const original = prototype[name];
                    prototype[name] = function(...args) {
                        profile.gl[name] = (profile.gl[name] || 0) + 1;
                        return original.apply(this, args);
                    };
                }
            }
        });
        await page.route(scriptURL, route => route.fulfill({body: library, contentType: 'application/javascript'}));
        await page.goto(`file://${path.join(temp, 'grid.html')}`);
        await page.waitForFunction(name => window[name]?.models.length, viewerName);
        await page.evaluate(name => {
            window.currentProfileViewer = window[name];
            window.oldProfileGrids = [];
            for (const viewer of currentProfileViewer.viewers.flat()) {
                const setFrame = viewer.setFrame;
                viewer.setFrame = function(...args) { profile.frames++; return setFrame.apply(this, args); };
            }
        }, viewerName);
        const cdp = await page.context().newCDPSession(page);
        await cdp.send('Performance.enable');
        const snapshot = async phase => {
            const value = await page.evaluate(() => {
                const viewers = currentProfileViewer.viewers.flat(), model = viewers[0].models[0];
                const callbacks = new Set([viewers, ...oldProfileGrids.map(grid => grid.flat())]
                    .flat().flatMap(v => [...v.animationTimers].map(timer => timer.fn)));
                return {ms: performance.now(), gl: {...profile.gl}, frameCalls: profile.frames,
                    pendingCallbacks: [...profile.pending.values()].filter(fn => callbacks.has(fn)).length,
                    registeredTimers: viewers.reduce((n, v) => n + v.animationTimers.size, 0),
                    viewers: viewers.length, models: viewers.map(v => v.models.length),
                    frames: model.frames.length, geometryCounter: viewers[0].renderer.info.memory.geometries,
                    pickingShapes: model.frames.reduce((n, frame) => n + frame.reduce((n, atom) =>
                        n + Object.values(atom.intersectionShape || {}).reduce((n, shapes) => n + shapes.length, 0), 0), 0),
                    oldTimers: oldProfileGrids.map(grid => grid.flat().reduce((n, v) => n + v.animationTimers.size, 0))};
            });
            value.phase = phase;
            value.metrics = (await cdp.send('Performance.getMetrics')).metrics.filter(item =>
                ['TaskDuration', 'JSHeapUsedSize', 'Nodes'].includes(item.name));
            for (const expression of ['window', 'document.body']) {
                const {result} = await cdp.send('Runtime.evaluate', {expression});
                const {listeners} = await cdp.send('DOMDebugger.getEventListeners', {objectId: result.objectId});
                value[expression] = listeners.reduce((counts, item) =>
                    (counts[item.type] = (counts[item.type] || 0) + 1, counts), {});
            }
            console.log(JSON.stringify(value));
        };
        console.log(JSON.stringify({browser: browser.version(), scriptURL,
            guardInstalled: html.includes('_tooltoadVibrationCleanup'),
            injectedGuard: flags.includes('--fixed')}));
        await snapshot('fresh');
        for (let elapsed = 0; elapsed < Number(seconds); elapsed += 30) {
            await page.waitForTimeout(Math.min(30, Number(seconds) - elapsed) * 1000);
            await snapshot('playback');
        }
        await page.evaluate(() => { for (let i = 0; i < 3; i++) window.dispatchEvent(new Event('resize')); });
        await page.waitForTimeout(1000);
        await snapshot('three-resizes');
        for (let iteration = 0; iteration < 3; iteration++) {
            await page.evaluate(({html, viewerName}) => {
                oldProfileGrids.push(currentProfileViewer.viewers);
                document.querySelector('div[id^="3dmolviewer_"]').remove();
                const section = document.createElement('section'); section.innerHTML = html;
                document.body.append(section);
                section.querySelectorAll('script').forEach(script => (0, eval)(script.textContent));
                $3Dmolpromise.then(() => { window.currentProfileViewer = window[viewerName]; });
            }, {html, viewerName});
            await page.waitForTimeout(2000);
            await snapshot(`replacement-${iteration + 1}`);
        }
    } finally {
        await browser.close();
        fs.rmSync(temp, {recursive: true, force: true});
    }
})().catch(error => { console.error(error); process.exitCode = 1; });
