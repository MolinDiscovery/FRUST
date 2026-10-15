// Verify actual timer cadence with a cached older engine and a fresh engine.
// Usage: NODE_PATH=/path/to/node_modules node check_cached_engine.cjs grid.html
const {chromium} = require('playwright');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');

(async () => {
    const input = path.resolve(process.argv[2]);
    const html = fs.readFileSync(input, 'utf8');
    const viewerName = html.match(/var (viewer_\d+) = null/)[1];
    const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
    const versions = ['https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js',
        'https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js'];
    const libraries = await Promise.all(versions.map(async url => {
        const response = await fetch(url);
        if (!response.ok) throw Error(`Fetch failed: ${url}`);
        return response.text();
    }));
    const browser = await chromium.launch({headless: true,
        executablePath: process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        args: ['--disable-background-timer-throttling']});
    try {
        for (const cachedOld of [true, false]) {
            const page = await browser.newPage({viewport: {width: 1560, height: 950}, deviceScaleFactor: 2});
            let requestedNew = 0;
            await page.addInitScript(() => {
                window.pending = new Map();
                const set = window.setTimeout, clear = window.clearTimeout;
                window.setTimeout = function(fn, ms, ...args) {
                    if (typeof fn !== 'function') return set(fn, ms, ...args);
                    const id = set(() => { pending.delete(id); fn(...args); }, ms);
                    pending.set(id, fn);
                    return id;
                };
                window.clearTimeout = function(id) { pending.delete(id); return clear(id); };
            });
            await page.route('**/3Dmol-min.js', route => {
                requestedNew++;
                return route.fulfill({body: libraries[1], contentType: 'application/javascript'});
            });
            if (cachedOld) {
                await page.goto('about:blank');
                await page.addScriptTag({content: libraries[0]});
                await page.evaluate(() => { window.$3Dmolpromise = Promise.resolve(); });
                await page.setContent(html.replace(/<script>[\s\S]*?<\/script>/g, ''));
                await page.evaluate(script => (0, eval)(script), script);
            } else {
                await page.goto(pathToFileURL(input).href);
            }
            await page.waitForFunction(name => window[name]?.models.length, viewerName);
            await page.evaluate(name => {
                window.v = window[name];
                window.framesRun = 0;
                for (const viewer of v.viewers.flat()) {
                    const setFrame = viewer.setFrame;
                    viewer.setFrame = function(...args) { framesRun++; return setFrame.apply(this, args); };
                }
            }, viewerName);
            const initial = await page.evaluate(() => ({time: performance.now(), frames: framesRun}));
            await page.waitForTimeout(5000);
            const result = await page.evaluate(initial => {
                const viewers = v.viewers.flat().filter(v => v.models.length);
                const callbacks = new Set(viewers.flatMap(v => [...v.animationTimers].map(t => t.fn)));
                return {viewers: viewers.length,
                    callbacks: [...pending.values()].filter(fn => callbacks.has(fn)).length,
                    perViewerFps: (framesRun - initial.frames) / (performance.now() - initial.time) * 1000 / viewers.length};
            }, initial);
            console.log(JSON.stringify({cachedOld, requestedNew, ...result}));
            if (result.callbacks !== result.viewers || result.perViewerFps > 25 || result.perViewerFps <= 0) {
                throw Error('Unexpected animation callback count or cadence');
            }
            await page.close();
        }
    } finally {
        await browser.close();
    }
})().catch(error => { console.error(error); process.exitCode = 1; });
