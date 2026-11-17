const assert = require("node:assert/strict");
const { mkdir, readFile, writeFile } = require("node:fs/promises");
const path = require("node:path");
const vm = require("node:vm");

function buildDemoHarness(audioFiles) {
  const elements = new Map();
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
      classList: { toggle() {} },
      addEventListener(name, callback) { this.events[name] = callback; },
      pause() { this.paused = true; },
      getContext() { return drawing; },
      replaceChildren(...children) { this.children = children; }
    };
  }

  // Use fixed draws for one reproducible permutation of the browser demo.
  const draws = [0.7, 0.8, 0.9];
  const math = Object.create(Math);
  math.random = () => draws.shift();
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
  return { elements, sandbox };
}

async function main() {
  const page = await readFile(path.join(__dirname, "birdsong_demo.html"), "utf8");
  const script = page.match(/<script>([\s\S]+)<\/script>/)?.[1];
  assert(script, "Expected the demo script in birdsong_demo.html");

  const audioFiles = [];
  const { elements, sandbox } = buildDemoHarness(audioFiles);
  vm.runInNewContext(script, sandbox, { timeout: 5000 });
  const labels = selector => elements.get(selector).children.map(item => item.textContent);
  const original = labels("#original");
  assert.deepEqual(original, ["UP", "FLAT", "DOWN", "WIGGLE"]);
  assert.deepEqual(labels("#current"), original);

  elements.get("#shuffle").events.click();
  const shuffled = labels("#current");
  assert.deepEqual(shuffled, ["UP", "FLAT", "WIGGLE", "DOWN"]);
  assert.match(elements.get("#comparison").textContent, /^Edit distance: 2\./);

  elements.get("#reset").events.click();
  assert.deepEqual(labels("#current"), original);
  assert.match(elements.get("#comparison").textContent, /^Edit distance: 0\./);

  const [originalWav, shuffledWav, resetWav] = await Promise.all(
    audioFiles.map(async file => Buffer.from(await file.arrayBuffer()))
  );
  assert.equal(originalWav.toString("ascii", 0, 4), "RIFF");
  assert.equal(originalWav.toString("ascii", 8, 12), "WAVE");
  assert.equal(originalWav.readUInt32LE(24), 16000);
  assert.equal(shuffledWav.length, originalWav.length);
  assert.notDeepEqual(shuffledWav, originalWav);
  assert.deepEqual(resetWav, originalWav);

  console.log("Original: " + original.join(" "));
  console.log("Shuffled: " + shuffled.join(" "));
  console.log("Edit distance: 2");
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

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
