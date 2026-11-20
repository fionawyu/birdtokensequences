const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { mkdir, writeFile } = require("node:fs/promises");
const path = require("node:path");
const vm = require("node:vm");

// Runs the page's own <script> in a minimal fake DOM, so these checks test the
// exact code the browser runs.
function loadDemo() {
  const page = readFileSync(path.join(__dirname, "birdsong_demo.html"), "utf8");
  const script = page.match(/<script>([\s\S]+)<\/script>/)?.[1];
  assert(script, "Expected the demo script in birdsong_demo.html");

  const elements = new Map();
  const audioFiles = [];
  const drawing = {
    fillRect() {},
    fillText() {},
    drawImage() {},
    putImageData(image) {
      assert(image.data.some((value, index) => index % 4 === 0 && value > 20),
        "The spectrogram must contain signal above the background");
    }
  };

  function element() {
    return {
      width: 720,
      height: 285,
      events: {},
      currentTime: 0,
      paused: true,
      hidden: false,
      disabled: false,
      textContent: "",
      children: [],
      classList: { toggle() {} },
      addEventListener(name, callback) { this.events[name] = callback; },
      pause() { this.paused = true; },
      getContext() { return drawing; },
      append(...children) { this.children.push(...children); },
      replaceChildren(...children) { this.children = children; }
    };
  }

  // Random draws are queued by each test, so every click is reproducible.
  const draws = [];
  const math = Object.create(Math);
  math.random = () => {
    assert(draws.length > 0, "The test did not queue a random draw for this click");
    return draws.shift();
  };
  const sandbox = {
    Blob,
    Math: math,
    ImageData: class {
      constructor(width, height) {
        Object.assign(this, { width, height, data: new Uint8ClampedArray(width * height * 4) });
      }
    },
    document: {
      createElement: element,
      querySelector(selector) {
        if (!elements.has(selector)) elements.set(selector, element());
        return elements.get(selector);
      }
    },
    URL: {
      revokeObjectURL() {},
      createObjectURL(blob) {
        audioFiles.push(blob);
        return "blob:demo-" + audioFiles.length;
      }
    }
  };
  vm.runInNewContext(script, sandbox, { timeout: 5000 });

  const el = selector => elements.get(selector);
  return {
    sandbox,
    el,
    audioFiles,
    labels: selector => el(selector).children.map(item => item.textContent),
    click(selector, ...queued) {
      draws.push(...queued);
      el(selector).events.click();
      assert.equal(draws.length, 0, "Every queued random draw should be used by " + selector);
    }
  };
}

async function toBuffer(blob) {
  return Buffer.from(await blob.arrayBuffer());
}

async function main() {
  const demo = loadDemo();
  const { el, labels, click, sandbox } = demo;
  const comparison = () => el("#comparison").textContent;

  // Start: the original and the current sequence match.
  const original = labels("#original");
  assert.deepEqual(original, ["UP", "FLAT", "DOWN", "WIGGLE"]);
  assert.deepEqual(labels("#current"), original);
  assert.match(comparison(), /^Edit distance: 0\./);
  assert.equal(el("#reset").disabled, true);

  // Duration: the text and the axis both end at the real length of the audio.
  assert.match(el("#duration").textContent, /^Duration: 1\.74 s \(27840 samples at 16 kHz\)/);
  assert.deepEqual(Array.from(sandbox.timeTicks(1.74)), [0, .25, .5, .75, 1, 1.25, 1.5, 1.74]);
  assert.equal(sandbox.timeTicks(2.3).at(-1), 2.3);

  // Shuffle keeps the inventory, so the page says only the order changed.
  click("#shuffle", 0.7, 0.8, 0.9);
  const shuffled = labels("#current");
  assert.deepEqual(shuffled, ["UP", "FLAT", "WIGGLE", "DOWN"]);
  assert.match(comparison(), /^Edit distance: 2\. Same notes/);
  assert.match(comparison(), /only the order changed/);
  assert.equal(el("#reset").disabled, false);

  click("#reset");
  assert.deepEqual(labels("#current"), original);
  assert.match(comparison(), /^Edit distance: 0\./);

  // Adding a note changes the inventory; the page must not say "only the order".
  el("#noteType").value = "UP";
  click("#add", 0);
  assert.deepEqual(labels("#current"), ["UP", "UP", "FLAT", "DOWN", "WIGGLE"]);
  assert.match(comparison(), /^Edit distance: 1\. The notes differ from the original \(added 1× UP\)/);
  assert.doesNotMatch(comparison(), /only the order changed|Same notes/);
  assert.match(el("#lastAction").textContent, /^Added UP at position 1\./);

  // Removing a note: the edit distance is 2, the same as the shuffle above,
  // but the inventory differs, and the message says so.
  click("#remove", 0.99);
  assert.deepEqual(labels("#current"), ["UP", "UP", "FLAT", "DOWN"]);
  assert.match(comparison(), /^Edit distance: 2\. The notes differ from the original \(added 1× UP; removed 1× WIGGLE\)/);
  assert.match(el("#inventory").textContent, /Current notes: UP ×2, FLAT ×1, DOWN ×1\./);

  // A new original sequence (here three UPs and a FLAT) replaces what Reset returns to.
  click("#newOriginal", 0.25, 0, 0, 0, 0.3);
  const newOriginal = ["UP", "UP", "UP", "FLAT"];
  assert.deepEqual(labels("#original"), newOriginal);
  assert.deepEqual(labels("#current"), newOriginal);
  assert.match(comparison(), /^Edit distance: 0\./);

  click("#shuffle", 0, 0, 0);
  assert.deepEqual(labels("#current"), ["UP", "UP", "FLAT", "UP"]);
  assert.match(comparison(), /only the order changed/);
  click("#reset");
  assert.deepEqual(labels("#current"), newOriginal, "Reset returns to the new original, not the first one");

  // A sequence with one note type has no other order, so Shuffle is disabled.
  el("#customOriginal").value = "up up up";
  click("#useCustom");
  assert.deepEqual(labels("#original"), ["UP", "UP", "UP"]);
  assert.equal(el("#shuffle").disabled, true);
  assert.equal(el("#remove").disabled, false);

  // Invalid input is rejected and leaves the original unchanged.
  el("#customOriginal").value = "UP BANANA";
  click("#useCustom");
  assert.match(el("#lastAction").textContent, /^Unknown note: BANANA/);
  assert.deepEqual(labels("#original"), ["UP", "UP", "UP"]);

  // WAV files: format, length, and the reset audio equals the original audio.
  const [originalWav, shuffledWav, resetWav] = await Promise.all(demo.audioFiles.slice(0, 3).map(toBuffer));
  assert.equal(originalWav.toString("ascii", 0, 4), "RIFF");
  assert.equal(originalWav.toString("ascii", 8, 12), "WAVE");
  assert.equal(originalWav.readUInt32LE(24), 16000);
  assert.equal(originalWav.readUInt32LE(40), 2 * 27840, "data chunk holds 1.74 s of 16-bit samples");
  assert.equal(shuffledWav.length, originalWav.length);
  assert.notDeepEqual(shuffledWav, originalWav);
  assert.deepEqual(resetWav, originalWav);

  console.log("Original: " + original.join(" "));
  console.log("Shuffled: " + shuffled.join(" ") + " (edit distance 2, same notes)");
  console.log("Added and removed: UP UP FLAT DOWN (edit distance 2, different notes)");
  console.log("New original: " + newOriginal.join(" "));
  console.log("Duration: 1.74 s");
  if (process.argv[2]) {
    const folder = path.resolve(process.argv[2]);
    await mkdir(folder, { recursive: true });
    await Promise.all([
      writeFile(path.join(folder, "original.wav"), originalWav),
      writeFile(path.join(folder, "shuffled.wav"), shuffledWav)
    ]);
    console.log("WAV files: " + folder);
  }
}

// Render any sequence with the page's own synthesizer.
async function renderSequence(labels, file) {
  const { sandbox } = loadDemo();
  const phrase = sandbox.renderPhrase(labels);
  await writeFile(file, await toBuffer(sandbox.wav(phrase.signal)));
  return phrase.duration;
}

module.exports = { loadDemo, renderSequence };

if (require.main === module) {
  main().catch(error => {
    console.error(error);
    process.exitCode = 1;
  });
}
