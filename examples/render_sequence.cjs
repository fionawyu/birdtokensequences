// Write the WAV the demo page plays for a note sequence.
//   node examples/render_sequence.cjs "UP FLAT DOWN WIGGLE" out.wav
const { renderSequence } = require("./reproduce_demo.cjs");

const [text, file] = process.argv.slice(2);
if (!text || !file) {
  console.error('Usage: node examples/render_sequence.cjs "UP FLAT DOWN WIGGLE" out.wav');
  process.exit(2);
}
renderSequence(text.trim().split(/\s+/), file).then(duration => {
  console.log(file + " " + duration.toFixed(2) + " s");
}).catch(error => {
  console.error(error.message);
  process.exitCode = 1;
});
